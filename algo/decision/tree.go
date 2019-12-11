package decision

import (
	"fmt"
	"rf/algo"
	"rf/mathelper"

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

func (Tree Tree) IsFitted() bool {
	panic("not implemented !")
	return false
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
Parameters allowed are maxDepth and minSize
*/
func Fit(m *mat.Dense, yCol int, params map[string]int) algo.Model {
	return fit(m, yCol, params["maxDepth"], params["minSize"])
}
func fit(m *mat.Dense, yCol, maxDepth, minSize int, depth ...int) (tree *Tree) {
	col, threshold, score, l, r := bestSplit(m, yCol)
	tree = &Tree{
		Feature: col,
		Value:   threshold,
	}
	var d int = 1
	if len(depth) > 0 {
		d = depth[0]
	}

	if l == nil && r == nil {
		v := term(m, yCol)
		tree.Value = v
		tree.Feature = 0
		return
	}
	if l == nil || r == nil {
		concat := &mat.Dense{}
		concat.Stack(l, r)
		v := term(concat, yCol)
		tree.Left = &Tree{Value: v}
		tree.Right = &Tree{Value: v}
		return
	}
	if d >= maxDepth {
		tree.Left = &Tree{Value: term(l, yCol)}
		tree.Right = &Tree{Value: term(r, yCol)}
		return
	}
	lr, _ := l.Dims()
	if lr > minSize && score > 0 {
		tree.Left = fit(l, yCol, maxDepth, minSize, d+1)
	} else {
		tree.Left = &Tree{Value: term(l, yCol)}
	}

	rr, _ := l.Dims()
	if rr > minSize && score > 0 {
		tree.Right = fit(r, yCol, maxDepth, minSize, d+1)
	} else {
		tree.Right = &Tree{Value: term(r, yCol)}
	}

	return
}

func (t Tree) String() string {
	return printTree(&t)
}

func printTree(t *Tree, d ...int) string {
	s := ""
	depth := 0
	if len(d) > 0 {
		depth = d[0]
		for i := 0; i < depth; i++ {
			s += fmt.Sprint("\t")
		}
	}
	s += fmt.Sprint("[feature ", t.Feature, "; value ", t.Value, "] \n")
	if t.Left != nil {
		s += printTree(t.Left, depth+1)
	}
	if t.Right != nil {
		s += printTree(t.Right, depth+1)
	}
	return s
}

func split(m mat.Matrix, col int, threshold float64) (left, right *mat.Dense) {

	rowsCount, colCount := m.Dims()
	leftData, rirghtData := make([]float64, 0, rowsCount), make([]float64, 0, rowsCount)

	for i := 0; i < rowsCount; i++ {
		val := m.At(i, col)
		row := mat.Row(nil, i, m)

		if val < threshold {
			leftData = append(leftData, row...)
		} else {
			rirghtData = append(rirghtData, row...)
		}
	}
	if len(leftData) > 0 {
		left = mat.NewDense(len(leftData)/colCount, colCount, leftData)
	}
	if len(rirghtData) > 0 {
		right = mat.NewDense(len(rirghtData)/colCount, colCount, rirghtData)
	}

	return
}

// If yCol = -1, it takes the last column as y, else bestSplit takes m[yCol] as the label column
func bestSplit(m mat.Matrix, yCol int) (col int, threshold float64, score float64, left, right *mat.Dense) {
	col, threshold, score = 999, 999.0, 999.0

	dR, dC := m.Dims()
	if yCol == -1 {
		yCol = dC - 1
	}

	for j := 0; j < dC-1; j++ {
		for i := 0; i < dR; i++ {
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

func term(m *mat.Dense, yCol int) float64 {
	_, cl := m.Dims()
	if yCol == -1 {
		yCol = cl - 1
	}
	return mathelper.Mode(m.ColView(yCol))
}
