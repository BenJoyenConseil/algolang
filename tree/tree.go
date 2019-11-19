package tree

import (
	"gonum.org/v1/gonum/mat"
)

/*
Tree represents a decision Tree structure with Predict method
*/
type Tree struct {
	Left    *Tree
	Feature int
	Value   float64
	Right   *Tree
}

/*
Predict on a fitted Tree returns the corresponding class foreach row
*/
func (tree Tree) Predict(m *mat.Dense) (predictions []float64) {
	l, _ := m.Dims()
	for i := 0; i < l; i++ {
		predictions = append(predictions, tree.PredictRow(m.RowView(i)))

	}
	return
}

/*
PredictRow on a fitted Tree returns the corresponding class for a new unseen row
*/
func (tree Tree) PredictRow(row mat.Vector) float64 {
	if row.AtVec(tree.Feature) < tree.Value {
		if tree.Left != nil {
			return tree.Left.PredictRow(row)
		}
		return tree.Value
	}

	if tree.Right != nil {
		return tree.Right.PredictRow(row)
	}
	return tree.Value
}

/*
Fit builds and return a Tree fitted on data, and ready to predict new rows of []float64
*/
func Fit(m mat.Matrix, yCol, maxDepth, minSize int, depth ...int) (tree *Tree) {

	col, threshold, score, l, r := bestSplit(m, yCol)
	tree = &Tree{
		Feature: col,
		Value:   threshold,
	}

	var d int = 1
	if len(depth) > 0 {
		d = depth[0]
	}
	if l == nil || r == nil {
		concat := &mat.Dense{}
		concat.Stack(l, r)
		tree.Left = &Tree{Value: term(concat, yCol)}
		tree.Right = &Tree{Value: term(concat, yCol)}
		return
	}
	if d >= maxDepth {
		tree.Left = &Tree{Value: term(l, yCol)}
		tree.Right = &Tree{Value: term(r, yCol)}
		return
	}
	lr, _ := l.Dims()
	if lr > minSize && score > 0 {
		tree.Left = Fit(l, yCol, maxDepth, minSize, d+1)
	} else {
		tree.Left = &Tree{Value: term(l, yCol)}
	}

	rr, _ := l.Dims()
	if rr > minSize && score > 0 {
		tree.Right = Fit(r, yCol, maxDepth, minSize, d+1)
	} else {
		tree.Right = &Tree{Value: term(r, yCol)}
	}

	return
}

func split(m mat.Matrix, col int, threshold float64) (left, right *mat.Dense) {

	rowsCount, colCount := m.Dims()
	leftRows, rirghtRows := make([]float64, 0, rowsCount), make([]float64, 0, rowsCount)
	leftLen, rightLen := 0, 0

	for i := 0; i < rowsCount; i++ {
		val := m.At(i, col)
		row := mat.Row(nil, i, m)

		if val < threshold {
			leftRows = append(leftRows, row...)
			leftLen++
		} else {
			rirghtRows = append(rirghtRows, row...)
			rightLen++
		}
	}
	if leftLen != 0 {
		left = mat.NewDense(leftLen, colCount, leftRows)
	}
	if rightLen != 0 {
		right = mat.NewDense(rightLen, colCount, rirghtRows)
	}

	return
}

// If yCol = -1, it takes the last column as y, else bestSplit takes m[yCol] as the label column
func bestSplit(m mat.Matrix, yCol int) (col int, threshold float64, score float64, left, right *mat.Dense) {
	col, threshold, score = 999, 999.0, 999.0

	rl, cl := m.Dims()
	if yCol == -1 {
		yCol = cl - 1
	}

	for j := 0; j < cl-1; j++ {
		for i := 0; i < rl; i++ {
			l, r := split(m, j, m.At(i, j))
			if l == nil || r == nil {
				continue
			}
			gini := giniIndex(l.ColView(yCol), r.ColView(yCol))
			if gini < score {
				col, threshold, score, left, right = j, m.At(i, j), gini, l, r
			}
			if score == 0 {
				return
			}
		}
	}
	return
}

func giniIndex(leftY, rightY mat.Vector) (gini float64) {
	nSamples := leftY.Len() + rightY.Len()
	gini = 0.0

	for _, v := range []mat.Vector{leftY, rightY} {
		vSize := float64(v.Len())
		if vSize == 0 {
			continue
		}
		score := 0.0
		trues, falses := countClasses(v)
		pTrue := float64(trues) / vSize
		pFalse := float64(falses) / vSize
		score += pTrue*pTrue + pFalse*pFalse
		gini += (1.0 - score) * (vSize / float64(nSamples))
	}
	return
}
func countClasses(v mat.Vector) (trues, falses int) {
	isTrue := func(v float64) bool {
		if v == 1.0 {
			return true
		}
		return false
	}
	trues = vCount(isTrue, v)
	falses = v.Len() - trues
	return trues, falses
}

func vCount(f func(float64) bool, v mat.Vector) int {
	var n int

	for i := 0; i < v.Len(); i++ {
		if f(v.AtVec(i)) {
			n++
		}
	}
	return n
}

func mostCurrentValue(v mat.Vector) float64 {
	counter := make(map[float64]int)
	l, _ := v.Dims()
	for i := 0; i < l; i++ {
		counter[v.AtVec(i)]++
	}
	max := counter[v.AtVec(0)]
	maxOccurence := v.AtVec(0)
	for k, v := range counter {
		if v > max {
			max = v
			maxOccurence = k
		}
	}
	return maxOccurence
}

func term(m *mat.Dense, yCol int) float64 {
	_, cl := m.Dims()
	if yCol == -1 {
		yCol = cl - 1
	}
	return mostCurrentValue(m.ColView(yCol))
}
