package rf

type Serie []float64

/*
Dataset is float64 matrix
*/
type Dataset []Serie

/*
Tree represents a decision Tree structure with Predict method
*/
type Tree struct {
	Left      *Tree
	idFeature int
	Value     float64
	Right     *Tree
}

/*
Fit builds and return a Tree fitted on data, and ready to predict new rows of []float64
*/
func Fit(dataset Dataset, maxDepth, minSize int, depth ...int) (tree *Tree) {

	tree = new(Tree)
	var left, right Dataset
	var score float64
	tree.idFeature, tree.Value, score, left, right = bestSplit(dataset)

	var d int = 1
	if len(depth) > 0 {
		d = depth[0]
	}
	if left == nil || right == nil {
		tree.Left = &Tree{Value: term(append(left, right...))}
		tree.Right = &Tree{Value: term(append(left, right...))}
		return
	}

	if d >= maxDepth {
		tree.Left = &Tree{Value: term(left)}
		tree.Right = &Tree{Value: term(right)}
		return
	}
	if len(left) > minSize && score > 0 {
		tree.Left = Fit(left, maxDepth, minSize, d+1)
	} else {
		tree.Left = &Tree{Value: term(left)}
	}

	if len(right) > minSize && score > 0 {
		tree.Right = Fit(right, maxDepth, minSize, d+1)
	} else {
		tree.Right = &Tree{Value: term(right)}
	}

	return
}

/*
Predict on a fitted Tree returns the corresponding class for new unseen row
*/
func (tree *Tree) Predict(row []float64) float64 {
	if row[tree.idFeature] < tree.Value {
		if tree.Left != nil {
			return tree.Left.Predict(row)
		}
		return tree.Value
	} else {
		if tree.Right != nil {
			return tree.Right.Predict(row)
		}
		return tree.Value
	}
}

func giniIndex(left, right Dataset, classes []float64) (gini float64) {
	nSamples := len(left) + len(right)

	gini = 0.0
	rowGroups := []Dataset{left, right}
	for _, rows := range rowGroups {
		gSize := float64(len(rows))
		if gSize == 0.0 {
			continue
		}
		score := 0.0
		for _, c := range classes {
			classHit := 0.0
			for _, r := range rows {
				if y := r[len(r)-1]; y == c {
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

func split(dataset Dataset, idFeature int, threshold float64) (left, right Dataset) {

	for _, row := range dataset {
		if row[idFeature] < threshold {
			left = append(left, row)
		} else {
			right = append(right, row)
		}
	}
	return
}

func bestSplit(dataset Dataset) (idFeature int, threshold float64, score float64, left, right Dataset) {
	idFeature, threshold, score = 999, 999.0, 999.0
	classes := uniqueClass(dataset)

	for idCol := 0; idCol < len(dataset[0])-1; idCol++ {
		for _, row := range dataset {
			l, r := split(dataset, idCol, row[idCol])
			gini := giniIndex(l, r, classes)
			if gini < score {
				idFeature, threshold, score, left, right = idCol, row[idCol], gini, l, r
			}
			if score == 0.0 {
				return
			}
		}
	}
	return
}

func term(dataset Dataset) float64 {
	y := []float64{}
	for _, row := range dataset {
		y = append(y, row[len(row)-1])
	}
	return vote(y)
}

func uniqueClass(dataset Dataset) []float64 {
	m := map[float64]bool{}
	for _, row := range dataset {
		m[row[len(row)-1]] = true
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
