package ensemble

import (
	"fmt"
	"rf/ensemble/decision"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestSqrtRatio(t *testing.T) {
	// Given
	n := 10

	// When
	r := sqrtRatio(n)
	r2 := sqrtRatio(n) * float64(n)

	// Then
	assert.Equal(t, .3, r)
	assert.Equal(t, 3.0, r2)
}

func TestExtractFeatures(t *testing.T) {
	// Given
	m := mat.NewDense(10, 5, []float64{
		2.771244718, 1.784783929, 1.784783929, 0.0, 2.771244718,
		1.728571309, 1.169761413, 1.169761413, 0.0, 1.728571309,
		3.678319846, 2.81281357, 2.81281357, 0.0, 3.678319846,
		3.961043357, 2.61995032, 2.61995032, 0.0, 3.961043357,
		2.999208922, 2.209014212, 2.209014212, 0.0, 2.999208922,
		7.497545867, 3.162953546, 3.162953546, 1.0, 7.497545867,
		9.00220326, 3.339047188, 3.339047188, 1.0, 9.00220326,
		7.444542326, 0.476683375, 0.476683375, 1.0, 7.444542326,
		10.12493903, 3.234550982, 3.234550982, 1.0, 10.12493903,
		6.642287351, 3.319983761, 3.319983761, 1.0, 6.642287351,
	})
	yCol := 3

	// When
	fe := extractFeatures(m, yCol)

	// Then
	assert.Contains(t, fe, 0, 1, 2, 4)
}

func TestFit(t *testing.T) {
	// Given
	m := mat.NewDense(10, 5, []float64{
		2.771244718, 1.784783929, 1.784783929, 0.0, 2.771244718,
		1.728571309, 1.169761413, 1.169761413, 0.0, 1.728571309,
		3.678319846, 2.81281357, 2.81281357, 0.0, 3.678319846,
		3.961043357, 2.61995032, 2.61995032, 0.0, 3.961043357,
		2.999208922, 2.209014212, 2.209014212, 0.0, 2.999208922,
		7.497545867, 3.162953546, 3.162953546, 1.0, 7.497545867,
		9.00220326, 3.339047188, 3.339047188, 1.0, 9.00220326,
		7.444542326, 0.476683375, 0.476683375, 1.0, 7.444542326,
		10.12493903, 3.234550982, 3.234550982, 1.0, 10.12493903,
		6.642287351, 3.319983761, 3.319983761, 1.0, 6.642287351,
	})
	yCol := 3
	nEstimators := 5
	maxDepth := 3
	minSampleSplit := 1
	seed := int64(123)

	// When
	r := Fit(m, yCol, nEstimators, maxDepth, minSampleSplit, seed)

	// Then
	assert.Len(t, r.treeBag, 5)
	// Tree 0
	assert.Equal(t, 6.642287351, r.treeBag[0].Value)
	assert.Equal(t, 0.0, r.treeBag[0].Left.Value)
	assert.Equal(t, 1.0, r.treeBag[0].Right.Value)
	// Tree 1
	assert.Equal(t, 6.642287351, r.treeBag[1].Value)
	assert.Equal(t, 0.0, r.treeBag[1].Left.Value)
	assert.Equal(t, 1.0, r.treeBag[1].Right.Value)
	// Tree 2
	assert.Equal(t, 6.642287351, r.treeBag[2].Value)
	assert.Equal(t, 0.0, r.treeBag[2].Left.Value)
	assert.Equal(t, 1.0, r.treeBag[2].Right.Value)
	// Tree 3
	assert.Equal(t, 6.642287351, r.treeBag[3].Value)
	assert.Equal(t, 0.0, r.treeBag[3].Left.Value)
	assert.Equal(t, 1.0, r.treeBag[3].Right.Value)
	// Tree 4
	assert.Equal(t, 6.642287351, r.treeBag[4].Value)
	assert.Equal(t, 0.0, r.treeBag[4].Left.Value)
	assert.Equal(t, 1.0, r.treeBag[4].Right.Value)
}

func TestRandomSubColumns(t *testing.T) {
	// Givne
	columns := []int{0, 1, 2, 3, 4, 5}
	seed := int64(123)

	// When
	r := randomSubColumns(columns, 4, seed)

	// Then
	assert.Len(t, r, 4)
	assert.NotContains(t, r, 4, 5)
	assert.Contains(t, r, 0, 1, 2, 3)
}

func TestIsFitted(t *testing.T) {
	// Given
	var model Model = &RandomForest{
		Score:   0.9,
		treeBag: []*decision.Tree{&decision.Tree{}},
	}

	// When
	r := model.IsFitted()

	// Then
	assert.True(t, r)
}

func TestSubSample(t *testing.T) {
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
	seed := int64(123)

	// When
	r := subsample(m, 0.4, []int{1, 2}, seed)

	// Then
	lr, lc := r.Dims()
	assert.Equal(t, []float64{3.162953546, 1.0}, []float64{r.At(0, 0), r.At(0, 1)})
	assert.Equal(t, []float64{3.319983761, 1.0}, []float64{r.At(1, 0), r.At(1, 1)})
	assert.Equal(t, []float64{2.61995032, 0.0}, []float64{r.At(2, 0), r.At(2, 1)})
	assert.Equal(t, 4, lr)
	assert.Equal(t, 2, lc)
}

func TestRand(t *testing.T) {

	m := mat.NewDense(10, 5, []float64{
		2.771244718, 1.784783929, 1.784783929, 0.0, 2.771244718,
		1.728571309, 1.169761413, 1.169761413, 0.0, 1.728571309,
		3.678319846, 2.81281357, 2.81281357, 0.0, 3.678319846,
		3.961043357, 2.61995032, 2.61995032, 0.0, 3.961043357,
		2.999208922, 2.209014212, 2.209014212, 0.0, 2.999208922,
		7.497545867, 3.162953546, 3.162953546, 1.0, 7.497545867,
		9.00220326, 3.339047188, 3.339047188, 1.0, 9.00220326,
		7.444542326, 0.476683375, 0.476683375, 1.0, 7.444542326,
		10.12493903, 3.234550982, 3.234550982, 1.0, 10.12493903,
		6.642287351, 3.319983761, 3.319983761, 1.0, 6.642287351,
	})
	_, dimC := m.Dims()
	ratioR := 1.0
	ratioC := sqrtRatio(dimC - 1)
	fmt.Println("ratio rows ", ratioR)
	fmt.Println("ratio columns ", ratioC)
	ycol := 3
	features := extractFeatures(m, ycol)
	fmt.Println("features :", features)
	rdmCols := randomSubColumns(features, ratioC, 123)
	fmt.Println("rdmCols :", rdmCols)
	sub := subsample(m, ratioR, append(rdmCols, ycol), 123)
	fmt.Print("sub matrix\n", mat.Formatted(sub), "\n")

	tree := decision.Fit(sub, 2, 2, 1)

	fmt.Println(tree)

	t.Fail()
}
