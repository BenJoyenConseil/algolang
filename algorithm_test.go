package rf

import (
	"fmt"
	"testing"
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
	if r != 0.3 {
		t.Error(r)
	}
	if r2 != 0.9 {
		t.Error(r2)
	}
	if r3 != 0.1 {
		t.Error(r3)
	}
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
	if r != 0.5 {
		t.Error(r)
	}

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
	if r != 0.4444444444444444 {
		t.Error(r)
	}

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
	if len(left) != 3 {
		t.Error(len(left))
	}
	if len(right) != 2 {
		t.Error(len(right))
	}
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
	if idFeature != 0 {
		t.Error("Wrong idFeature", idFeature)
	}
	if threshold != 6.642287351 {
		t.Error("Wrong threshold", threshold)
	}
	if score != 0.0 {
		t.Error("Wrong score", score)
	}
	if right[0][0] != 7.497545867 {
		t.Error("Wrong groups", right)
	}
}

func TestUniqueClass(t *testing.T) {
	// Given
	tab := Dataset{
		{1.0}, {1.0}, {0.0}, {0.0}, {2.1}, {2.2}, {0.0},
	}
	// When
	r := uniqueClass(tab)

	// Then
	if len(r) != 4 {
		t.Error("Wrong ! ", r, len(r), "!= 4")
	}
}

func TestTerm(t *testing.T) {
	// Given
	data := Dataset{{1.0}, {1.0}, {0.0}, {0.1}, {0.2}}

	// When
	r := term(data)

	// Then
	if r != 1.0 {
		t.Error("Wrong terminal value", r)
	}
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

	if r.Left.Value != 0 {
		t.Error(r.Left.Value, 0)
	}
	if r.Right.Value != 1 {
		t.Error(r.Right.Value, 1)
	}
	if r.idFeature != 0 {
		t.Error(r.idFeature, 0)
	}
	if r.Value != 6.642287351 {
		t.Error(r.Value, 6.642287351)
	}
}

func printTree(t *Tree, d ...int) string {
	s := "\n"
	depth := 1
	if len(d) < 0 {
		depth = d[0]
	}
	for d := 0; d < depth; d++ {
		s += fmt.Sprintf("a")
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
	if r != row[2] {
		t.Error(r, row[2])
	}
	// Then
	if r2 != row2[2] {
		t.Error(r2, row2[2])
	}
}
