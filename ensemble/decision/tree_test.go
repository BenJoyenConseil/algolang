package decision

import (
	"fmt"
	"rf/eval"
	"rf/io"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/tobgu/qframe/config/csv"
	"gonum.org/v1/gonum/mat"
)

func TestSplit(t *testing.T) {
	// Given
	m := mat.NewDense(5, 3, []float64{
		1.3, 9999.9, 0.4,
		1.3, 9999.9, 0.3,
		1.3, 9999.9, 0.4,
		1.3, 9999.9, 0.2,
		1.3, 9999.9, 0.1,
	})

	// When
	left, right := split(m, 2, 0.4)

	// Then
	lr, lc := left.Dims()
	rr, rc := right.Dims()
	assert.Equal(t, 3, lr)
	assert.Equal(t, 3, lc)
	assert.Equal(t, 2, rr)
	assert.Equal(t, 3, rc)
	assert.Equal(t, 0.1, left.At(2, 2))
	assert.Equal(t, 0.4, right.At(1, 2))
}

func TestBestSplit(t *testing.T) {
	// Given
	m := mat.NewDense(10, 3, []float64{
		2.771244718, 1.784783929, 0.0,
		1.728571309, 1.169761413, 0.0,
		3.678319846, 2.81281357, 0.0,
		3.961043357, 2.61995032, 0.0,
		2.999208922, 2.209014212, 0.0,
		7.497545867, 3.162953546, 1.0,
		9.00220326, 3.339047188, 1.0,
		7.444542326, 0.476683375, 1.0,
		10.12493903, 3.234550982, 1.0,
		6.642287351, 3.319983761, 1.0,
	})

	// When
	col, threshold, score, left, right := bestSplit(m, -1)

	// Then
	assert.Equal(t, 0, col)
	assert.Equal(t, 6.642287351, threshold)
	assert.Equal(t, 0.0, score)
	assert.Equal(t, 7.497545867, right.At(0, 0))
	assert.Equal(t, 0.0, left.At(4, 2))
	assert.Equal(t, 1.0, right.At(4, 2))
}
func TestBestSplit_Ycol(t *testing.T) {
	// Given
	m := mat.NewDense(10, 3, []float64{
		2.771244718, 0.0, 1.784783929,
		1.728571309, 0.0, 1.169761413,
		3.678319846, 0.0, 2.81281357,
		3.961043357, 0.0, 2.61995032,
		2.999208922, 0.0, 2.209014212,
		7.497545867, 1.0, 3.162953546,
		9.00220326, 1.0, 3.339047188,
		7.444542326, 1.0, 0.476683375,
		10.12493903, 1.0, 3.234550982,
		6.642287351, 1.0, 3.319983761,
	})

	// When
	col, threshold, score, left, right := bestSplit(m, 1)

	// Then
	assert.Equal(t, 0, col)
	assert.Equal(t, 6.642287351, threshold)
	assert.Equal(t, 0.0, score)
	assert.Equal(t, 7.497545867, right.At(0, 0))
	assert.Equal(t, 0.0, left.At(4, 1))
	assert.Equal(t, 1.0, right.At(4, 1))
}
func TestBestSplit_NilNil(t *testing.T) {
	// Given
	m := mat.NewDense(2, 3, []float64{
		0.21431, 0.87711, 1.0,
		0.21431, 0.87711, 1.0,
	})

	// When
	_, _, _, left, right := bestSplit(m, 1)

	// Then
	assert.NotNil(t, left)
	assert.NotNil(t, right)
}

func TestGiniIndex(t *testing.T) {
	// Given
	group1 := mat.NewVecDense(2, []float64{0.0, 1.0})
	group2 := mat.NewVecDense(2, []float64{1.0, 0.0})

	// When
	r := giniIndex(group1, group2)

	// Then
	assert.Equal(t, 0.5, r)

	// Given
	group1 = mat.NewVecDense(1, []float64{0.0})
	group2 = mat.NewVecDense(9, []float64{0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0})

	// When
	r = giniIndex(group1, group2)

	// Then
	assert.Equal(t, 0.4444444444444444, r)
}

func BenchmarkBestSplit(b *testing.B) {
	m := mat.NewDense(10, 3, []float64{
		2.771244718, 1.784783929, 0.0,
		1.728571309, 1.169761413, 0.0,
		3.678319846, 2.81281357, 0.0,
		3.961043357, 2.61995032, 0.0,
		2.999208922, 2.209014212, 0.0,
		7.497545867, 3.162953546, 1.0,
		9.00220326, 3.339047188, 1.0,
		7.444542326, 0.476683375, 1.0,
		10.12493903, 3.234550982, 1.0,
		6.642287351, 3.319983761, 1.0,
	})

	var feature int
	var threshold, score float64
	var left, right mat.Matrix

	for i := 0; i < b.N; i++ {
		feature, threshold, score, left, right = bestSplit(m, -1)
	}
	fmt.Println(feature, threshold, score, left, right)
}

func TestTerm(t *testing.T) {
	// Given
	data := mat.NewDense(9, 1, []float64{0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0})

	// When
	r := term(data, -1)

	// Then
	assert.Equal(t, 1.0, r)
}

func TestTerm_YCol(t *testing.T) {
	// Given
	data := mat.NewDense(10, 3, []float64{
		2.771244718, 1.784783929, 0.0,
		1.728571309, 1.169761413, 0.0,
		3.678319846, 2.81281357, 0.0,
		3.961043357, 2.61995032, 0.0,
		2.999208922, 2.209014212, 1.0,
		7.497545867, 3.162953546, 1.0,
		9.00220326, 3.339047188, 1.0,
		7.444542326, 0.476683375, 1.0,
		10.12493903, 3.234550982, 1.0,
		6.642287351, 3.319983761, 1.0,
	})

	// When
	r := term(data, 2)

	// Then
	assert.Equal(t, 1.0, r)
}

func TestFit(t *testing.T) {
	// Given
	m := mat.NewDense(10, 3, []float64{
		2.771244718, 1.784783929, 0.0,
		1.728571309, 1.169761413, 0.0,
		3.678319846, 2.81281357, 0.0,
		3.961043357, 2.61995032, 0.0,
		2.999208922, 2.209014212, 0.0,
		7.497545867, 3.162953546, 1.0,
		9.00220326, 3.339047188, 1.0,
		7.444542326, 0.476683375, 1.0,
		10.12493903, 3.234550982, 1.0,
		6.642287351, 3.319983761, 1.0,
	})

	// When
	r := Fit(m, -1, 10, 1)

	// Then
	assert.Equal(t, 0.0, r.Left.Value)
	assert.Equal(t, 1.0, r.Right.Value)
	assert.Equal(t, 0, r.Feature)
	assert.Equal(t, 6.642287351, r.Value)
}

func TestPredict(t *testing.T) {
	// Given
	tree := &Tree{
		Feature: 0,
		Right:   &Tree{Value: 1.0},
		Value:   6.642287351,
		Left:    &Tree{Value: 0.0},
	}
	m := mat.NewDense(2, 2, []float64{
		2.771244718, 1.784783929,
		9.00220326, 3.339047188,
	})

	// When
	r := tree.Predict(m)

	// Then
	assert.Exactly(t, []float64{0.0, 1.0}, r)
}

func TestFunctional(t *testing.T) {

	types := map[string]string{"y": "float"}
	df := io.LoadCsv("../../data/data_banknote_authentication.txt", csv.Headers([]string{"col_0", "col_1", "col_2", "col_3", "y"}), csv.Types(types))
	m := io.ToMatrix(df)

	model := Fit(m, -1, 5, 10)
	preds := model.Predict(m)
	y, _ := df.FloatView("y")
	a := eval.Accuracy(y.Slice(), preds)
	printTree(model)
	fmt.Println(a)

	assert.Greater(t, a, 97.0)
}

func BenchmarkFit(b *testing.B) {
	for i := 0; i < b.N; i++ {

		types := map[string]string{"y": "float"}
		df := io.LoadCsv("../../data/data_banknote_authentication.txt", csv.Headers([]string{"col_0", "col_1", "col_2", "col_3", "y"}), csv.Types(types))
		m := io.ToMatrix(df)
		_ = Fit(m, -1, 5, 10)
	}
}
