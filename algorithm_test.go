package rf

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
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
	group1 := Dataset{
		{0.0},
		{1.0},
	}
	group2 := Dataset{
		{1.0},
		{0.0},
	}
	classes := []float64{1.0, 0.0}

	// When
	r := giniIndex(group1, group2, classes)

	// Then
	assert.Equal(t, 0.5, r)

	// Given
	group1 = Dataset{
		{0.0},
	}
	group2 = Dataset{
		{0.0},
		{0.0},
		{0.0},
		{0.0},
		{1.0},
		{1.0},
		{1.0},
		{1.0},
		{1.0},
	}

	// When
	r = giniIndex(group1, group2, classes)

	// Then
	assert.Equal(t, 0.4444444444444444, r)
}

func TestSplit(t *testing.T) {
	// Given
	dataset := Dataset{
		{0.3, 9999.9, 0.4},
		{0.3, 9999.9, 0.3},
		{0.3, 9999.9, 0.4},
		{0.3, 9999.9, 0.2},
		{0.3, 9999.9, 0.1},
	}

	// When
	left, right := split(dataset, 2, 0.4)

	// Then
	assert.Len(t, left, 3)
	assert.Len(t, right, 2)
}

func TestBestSplit(t *testing.T) {
	// Given
	dataset := Dataset{
		{2.771244718, 1.784783929, 0.0},
		{1.728571309, 1.169761413, 0.0},
		{3.678319846, 2.81281357, 0.0},
		{3.961043357, 2.61995032, 0.0},
		{2.999208922, 2.209014212, 0.0},
		{7.497545867, 3.162953546, 1.0},
		{9.00220326, 3.339047188, 1.0},
		{7.444542326, 0.476683375, 1.0},
		{10.12493903, 3.234550982, 1.0},
		{6.642287351, 3.319983761, 1.0},
	}

	// When
	idFeature, threshold, score, _, right := bestSplit(dataset)

	// Then
	assert.Equal(t, 0, idFeature)
	assert.Equal(t, 6.642287351, threshold)
	assert.Equal(t, 0.0, score)
	assert.Equal(t, 7.497545867, right[0][0])
}

func TestUniqueClass(t *testing.T) {
	// Given
	tab := Dataset{
		{1.0}, {1.0}, {0.0}, {0.0}, {2.1}, {2.2}, {0.0},
	}
	// When
	r := uniqueClass(tab)

	// Then
	assert.ElementsMatch(t, []float64{1.0, 0.0, 2.1, 2.2}, r)
}

func TestTerm(t *testing.T) {
	// Given
	data := Dataset{{1.0}, {1.0}, {0.0}, {0.1}, {0.2}}

	// When
	r := term(data)

	// Then
	assert.Equal(t, 1.0, r)
}

func TestFit(t *testing.T) {
	// Given
	data := Dataset{
		{2.771244718, 1.784783929, 0},
		{1.728571309, 1.169761413, 0},
		{3.678319846, 2.81281357, 0},
		{3.961043357, 2.61995032, 0},
		{2.999208922, 2.209014212, 0},
		{7.497545867, 3.162953546, 1},
		{9.00220326, 3.339047188, 1},
		{7.444542326, 0.476683375, 1},
		{10.12493903, 3.234550982, 1},
		{6.642287351, 3.319983761, 1},
	}

	// When
	r := Fit(data, 10, 1)

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
	row := []float64{2.771244718, 1.784783929, 0.0}
	row2 := []float64{9.00220326, 3.339047188, 1.0}

	// When
	r := tree.Predict(row[:1])
	r2 := tree.Predict(row2[:1])

	// Then
	assert.Equal(t, 0.0, r)
	assert.Equal(t, 1.0, r2)
}

func TestFunctional(t *testing.T) {

	dataset := loadCsv("./data_banknote_authentication.txt")
	splitTrainSize := int(len(dataset) / 3)

	model := Fit(dataset[:], 5, 10)
	t.Log(printTree(model))

	y, preds := []float64{}, []float64{}
	for _, row := range dataset[splitTrainSize:] {
		pred := model.Predict(row[:len(row)-1])
		row = append(row, pred)
		y = append(y, row[len(row)-2])
		preds = append(preds, row[len(row)-1])
	}
	a := Accuracy(y, preds)

	assert.Greater(t, a, 97.0)
}

func loadCsv(filename string) Dataset {
	dataset := Dataset{}
	csvFile, err := os.Open(filename)
	if err != nil {
		panic(err)
	}
	reader := csv.NewReader(bufio.NewReader(csvFile))
	for {
		row := []float64{}
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		for _, field := range record {
			if value, err := strconv.ParseFloat(field, 64); err == nil {
				row = append(row, value)
			}
		}
		dataset = append(dataset, row)
	}
	return dataset
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
