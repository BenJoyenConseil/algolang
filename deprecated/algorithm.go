package deprecated

// Model must be fitted, and it has a Predict method
type Model interface {
	Predict(DataFrame) Serie
}

/*
Tree represents a decision Tree structure with Predict method
*/
type Tree struct {
	Left      *Tree
	Feature   string
	idFeature int
	Value     float64
	Right     *Tree
}

/*
Fit builds and return a Tree fitted on data, and ready to predict new rows of []float64
*/
func Fit(df DataFrame, maxDepth, minSize int, depth ...int) (tree *Tree) {

	if _, ok := df["y"]; !ok {
		panic("The df DataFrame needs at least a \"y\" column to fit")
	}

	tree = new(Tree)
	var left, right DataFrame
	var score float64
	tree.idFeature, tree.Feature, tree.Value, score, left, right = bestSplit(df)

	var d int = 1
	if len(depth) > 0 {
		d = depth[0]
	}
	if left == nil || right == nil {
		tree.Left = &Tree{Value: term(Concat(left, right))}
		tree.Right = &Tree{Value: term(Concat(left, right))}
		return
	}

	if d >= maxDepth {
		tree.Left = &Tree{Value: term(left)}
		tree.Right = &Tree{Value: term(right)}
		return
	}
	if left.Size() > minSize && score > 0 {
		tree.Left = Fit(left, maxDepth, minSize, d+1)
	} else {
		tree.Left = &Tree{Value: term(left)}
	}

	if right.Size() > minSize && score > 0 {
		tree.Right = Fit(right, maxDepth, minSize, d+1)
	} else {
		tree.Right = &Tree{Value: term(right)}
	}

	return
}

/*
Predict on a fitted Tree returns the corresponding class foreach row
*/
func (tree Tree) Predict(df DataFrame) Serie {
	preds := Serie{}
	for i := 0; i < len(df[df.SortKeys()[0]]); i++ {
		preds = append(preds, tree.PredictRow(df.ILoc(i)))

	}
	return preds
}

/*
PredictRow on a fitted Tree returns the corresponding class for a new unseen row
*/
func (tree Tree) PredictRow(row []float64) float64 {
	if row[tree.idFeature] < tree.Value {
		if tree.Left != nil {
			return tree.Left.PredictRow(row)
		}
		return tree.Value
	} else {
		if tree.Right != nil {
			return tree.Right.PredictRow(row)
		}
		return tree.Value
	}
}

func giniIndex(left, right DataFrame, classes []float64) (gini float64) {
	nSamples := left.Size() + right.Size()
	gini = 0.0
	rowGroups := []Serie{left["y"], right["y"]}
	for _, rows := range rowGroups {
		gSize := float64(len(rows))
		if gSize == 0.0 {
			continue
		}
		score := 0.0
		for _, c := range classes {
			classHit := 0.0
			for _, val := range rows {
				if y := val; y == c {
					classHit++
				}
			}
			p := classHit / gSize
			score += p * p
		}
		gini += (1.0 - score) * (gSize / float64(nSamples))
	}
	return
}

func split(df DataFrame, feature string, threshold float64) (left, right DataFrame) {

	left, right = NewDataFrame(), NewDataFrame()
	for col := range df {
		left[col], right[col] = Serie{}, Serie{}
	}
	for index, val := range df[feature] {
		if val < threshold {
			left = left.AddRow(df.ILoc(index))
		} else {
			right = right.AddRow(df.ILoc(index))
		}
	}
	return
}

func bestSplit(df DataFrame) (idFeature int, feature string, threshold float64, score float64, left, right DataFrame) {
	idFeature, feature, threshold, score = 999, "x999", 999.0, 999.0
	classes := uniqueClass(df["y"])

	iCol := 0
	for _, col := range df.Drop("y").SortKeys() {
		for _, v := range df[col] {
			l, r := split(df, col, v)
			gini := giniIndex(l, r, classes)

			if gini < score {
				idFeature, feature, threshold, score, left, right = iCol, col, v, gini, l, r
			}
			if score == 0.0 {
				return
			}
		}
		iCol++
	}
	return
}

func term(df DataFrame) float64 {
	return vote(df["y"])
}

func uniqueClass(s Serie) []float64 {
	m := map[float64]bool{}
	for _, val := range s {
		m[val] = true
	}
	classes := []float64{}
	for k := range m {
		classes = append(classes, k)
	}
	return classes
}

func vote(predictions []float64) float64 {
	mapPred := make(map[float64]int)
	for _, p := range predictions {
		if val, ok := mapPred[p]; ok {
			mapPred[p] = val + 1
		} else {
			mapPred[p] = 1
		}
	}
	maxV := mapPred[predictions[0]]
	maxIndex := predictions[0]
	for k, v := range mapPred {
		if v > maxV {
			maxV = v
			maxIndex = k
		}
	}
	return maxIndex
}
