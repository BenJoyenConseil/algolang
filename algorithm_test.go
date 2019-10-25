package rf

import (
	"fmt"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestVote(t *testing.T) {
	// Given
	predictions := []float64{0.3, 0.4, 0.9, 0.5, 0.3}
	predictions2 := []float64{0.4, 0.4, 0.9, 0.9, 0.9}
	predictions3 := []float64{0.1, 0.4, 0.8}

	// When
	r := vote(predictions)
	r2 := vote(predictions2)
	r3 := vote(predictions3)

	// Then
	assert.Equal(t, 0.3, r)
	assert.Equal(t, 0.9, r2)
	assert.Equal(t, 0.1, r3)
}

func TestGiniIndex(t *testing.T) {
	// Given
	group1 := DataFrame{
		"y": {0.0, 1.0},
	}
	group2 := DataFrame{
		"y": {1.0, 0.0},
	}
	classes := Serie{1.0, 0.0}

	// When
	r := giniIndex(group1, group2, classes)

	// Then
	assert.Equal(t, 0.5, r)

	// Given
	group1 = DataFrame{
		"y": {0.0},
	}
	group2 = DataFrame{
		"y": {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0},
	}

	// When
	r = giniIndex(group1, group2, classes)

	// Then
	assert.Equal(t, 0.4444444444444444, r)
}

func TestSplit(t *testing.T) {
	// Given
	df := DataFrame{
		"x0": {1.3, 1.3, 1.3, 1.3, 1.3},
		"x1": {9999.9, 9999.9, 9999.9, 9999.9, 9999.9},
		"x2": {0.4, 0.3, 0.4, 0.2, 0.1},
	}

	// When
	left, right := split(df, "x2", 0.4)

	// Then
	assert.Equal(t, left.Size(), 3)
	assert.Equal(t, right.Size(), 2)
}

func TestBestSplit(t *testing.T) {
	// Given
	df := DataFrame{
		"x0": {2.771244718, 1.728571309, 3.678319846, 3.961043357, 2.999208922, 7.497545867, 9.00220326, 7.444542326, 10.12493903, 6.642287351},
		"x1": {1.784783929, 1.169761413, 2.81281357, 2.61995032, 2.209014212, 3.162953546, 3.339047188, 0.476683375, 3.234550982, 3.319983761},
		"y":  {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0},
	}

	// When
	idFeature, feature, threshold, score, _, right := bestSplit(df)

	// Then
	assert.Equal(t, 0, idFeature)
	assert.Equal(t, "x0", feature)
	assert.Equal(t, 6.642287351, threshold)
	assert.Equal(t, 0.0, score)
	assert.Equal(t, 7.497545867, right["x0"][0])
}

func TestUniqueClass(t *testing.T) {
	// Given
	tab := Serie{1.0, 1.0, 0.0, 0.0, 2.1, 2.2, 0.0}
	// When
	r := uniqueClass(tab)

	// Then
	assert.ElementsMatch(t, []float64{1.0, 0.0, 2.1, 2.2}, r)
}

func TestTerm(t *testing.T) {
	// Given
	data := DataFrame{
		"y": {1.0, 1.0, 0.0, 0.1, 0.2},
	}

	// When
	r := term(data)

	// Then
	assert.Equal(t, 1.0, r)
}

func TestFit(t *testing.T) {
	// Given
	df := DataFrame{
		"x0": {2.771244718, 1.728571309, 3.678319846, 3.961043357, 2.999208922, 7.497545867, 9.00220326, 7.444542326, 10.12493903, 6.642287351},
		"x1": {1.784783929, 1.169761413, 2.81281357, 2.61995032, 2.209014212, 3.162953546, 3.339047188, 0.476683375, 3.234550982, 3.319983761},
		"y":  {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0},
	}

	// When
	r := Fit(df, 10, 1)

	// Then
	assert.Equal(t, 0.0, r.Left.Value)
	assert.Equal(t, 1.0, r.Right.Value)
	assert.Equal(t, 0, r.idFeature)
	assert.Equal(t, 6.642287351, r.Value)
}

func TestPredict(t *testing.T) {
	// Given
	tree := &Tree{
		idFeature: 0,
		Right:     &Tree{Value: 1.0},
		Value:     6.642287351,
		Left:      &Tree{Value: 0.0},
	}
	data := Dataset{
		{2.771244718, 1.784783929, 0.0},
		{9.00220326, 3.339047188, 1.0},
	}

	// When
	r := tree.Predict(data)

	// Then
	assert.Exactly(t, Serie{0.0, 1.0}, r)
}

func TestFunctional(t *testing.T) {

	df := loadCsv("./data_banknote_authentication.txt")
	//splitTrainSize := int(df.Size() / 3)
	df["y"] = df["x4"]
	df = df.Drop("x4")

	model := Fit(df, 5, 10)
	t.Log(printTree(model))

	//a := Accuracy(y, preds)

	//assert.Greater(t, a, 97.0)
}

func loadCsv(filename string) DataFrame {
	csvFile, err := os.Open(filename)
	if err != nil {
		panic(err)
	}

	return ReadCSV(csvFile)
}

func printTree(t *Tree, d ...int) string {
	s := "\n"
	depth := 0
	if len(d) < 0 {
		depth = d[0]
	}
	for i := 0; i < depth; i++ {
		s += fmt.Sprintf("Node #%v", i)
	}
	s += fmt.Sprintf("X%v = %v", t.idFeature, t.Value)
	if t.Left != nil {
		s += printTree(t.Left, depth+1)
	}
	if t.Right != nil {
		s += printTree(t.Right, depth+1)
	}
	return s
}
