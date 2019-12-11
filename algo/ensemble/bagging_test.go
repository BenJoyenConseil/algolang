package ensemble

import (
	"fmt"
	"math/rand"
	"rf/algo/decision"
	"rf/mathelper"
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
	maxDepth := 1
	minSampleSplit := 1
	rand.Seed(1234)

	// When
	r := fit(m, yCol, nEstimators, maxDepth, minSampleSplit)

	// Then
	assert.Len(t, r.estimators, 5)
	// Tree 0
	assert.Equal(t, 6.642287351, r.estimators[0].Value)
	assert.Equal(t, 0.0, r.estimators[0].Left.Value)
	assert.Equal(t, 1.0, r.estimators[0].Right.Value)
	assert.Equal(t, 0, r.estimators[0].Feature)
	assert.Contains(t, r.feMapping[r.estimators[0]], 0, 1)
	// Tree 1
	assert.Equal(t, 6.642287351, r.estimators[1].Value)
	assert.Equal(t, 0.0, r.estimators[1].Left.Value)
	assert.Equal(t, 1.0, r.estimators[1].Right.Value)
	assert.Equal(t, 0, r.estimators[1].Feature)
	assert.Contains(t, r.feMapping[r.estimators[1]], 0, 1)
	// Tree 2
	assert.Equal(t, 7.444542326, r.estimators[2].Value)
	assert.Equal(t, 0.0, r.estimators[2].Left.Value)
	assert.Equal(t, 1.0, r.estimators[2].Right.Value)
	assert.Equal(t, 0, r.estimators[2].Feature)
	assert.Contains(t, r.feMapping[r.estimators[2]], 0, 1)
	// Tree 3
	assert.Equal(t, 3.234550982, r.estimators[3].Value)
	assert.Equal(t, 0.0, r.estimators[3].Left.Value)
	assert.Equal(t, 1.0, r.estimators[3].Right.Value)
	assert.Equal(t, 0, r.estimators[3].Feature)
	assert.Contains(t, r.feMapping[r.estimators[3]], 1, 2)
	// Tree 4
	assert.Equal(t, 3.162953546, r.estimators[4].Value)
	assert.Equal(t, 0.0, r.estimators[4].Left.Value)
	assert.Equal(t, 1.0, r.estimators[4].Right.Value)
	assert.Equal(t, 0, r.estimators[4].Feature)
	assert.Contains(t, r.feMapping[r.estimators[4]], 1, 2)
}

func TestPredict(t *testing.T) {
	// Given
	SIGNAL_NULL := .0
	rows := mat.NewDense(2, 4, []float64{
		SIGNAL_NULL, SIGNAL_NULL, 3.319983761, 6.642287351,
		SIGNAL_NULL, SIGNAL_NULL, 2.209014212, 2.999208922,
	})
	dtree1 := &decision.Tree{
		Value:   6.642287351,
		Feature: 0,
		Left:    &decision.Tree{Value: 0.0},
		Right:   &decision.Tree{Value: 1.0},
	}
	dtree2 := &decision.Tree{
		Value:   0,
		Feature: 1,
		Left:    &decision.Tree{Value: 1.0},
		Right:   &decision.Tree{Value: 0.0},
	}
	rf := &RandomForest{
		feMapping:  make(map[*decision.Tree][]int, 2),
		estimators: []*decision.Tree{dtree1, dtree2},
	}
	rf.feMapping[dtree2] = []int{2, 3}
	rf.feMapping[dtree1] = []int{3, 2}

	// When
	preds := rf.Predict(rows)

	// Then
	assert.Equal(t, 1.0, preds[0])
	assert.Equal(t, 0.0, preds[1])
}

func TestPredictRow(t *testing.T) {
	// Given
	SIGNAL_NULL := .0
	row := mathelper.Row{SIGNAL_NULL, SIGNAL_NULL, 3.319983761, 6.642287351}
	row2 := mathelper.Row{SIGNAL_NULL, SIGNAL_NULL, 2.209014212, 2.999208922}
	dtree1 := &decision.Tree{
		Value:   6.642287351,
		Feature: 0,
		Left:    &decision.Tree{Value: 0.0},
		Right:   &decision.Tree{Value: 1.0},
	}
	dtree2 := &decision.Tree{
		Value:   0,
		Feature: 1,
		Left:    &decision.Tree{Value: 1.0},
		Right:   &decision.Tree{Value: 0.0},
	}
	rf := &RandomForest{
		feMapping:  make(map[*decision.Tree][]int, 2),
		estimators: []*decision.Tree{dtree1, dtree2},
	}
	rf.feMapping[dtree2] = []int{2, 3}
	rf.feMapping[dtree1] = []int{3, 2}

	// When
	p := rf.PredictRow(row)
	p2 := rf.PredictRow(row2)

	// Then
	assert.Equal(t, 1.0, p)
	assert.Equal(t, 0.0, p2)
}

func TestRandomSubColumns(t *testing.T) {
	// Givne
	columns := []int{0, 1, 2, 3, 4, 5}
	rand.Seed(123)

	// When
	r := randomSubColumns(columns, 0.5)

	// Then
	fmt.Println(r)
	assert.Len(t, r, 3)
	assert.NotContains(t, r, 1)
	assert.NotContains(t, r, 2)
	assert.NotContains(t, r, 5)
	assert.Contains(t, r, 0, 0)
	assert.Contains(t, r, 0, 3)
	assert.Contains(t, r, 0, 4)
}

func TestIsFitted(t *testing.T) {
	// Given
	var model Model = &RandomForest{
		Score:      0.9,
		estimators: []*decision.Tree{&decision.Tree{}},
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
	rand.Seed(123)

	// When
	r := subsample(m, 0.4, []int{1, 2})

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
	rdmCols := randomSubColumns(features, ratioC)
	fmt.Println("rdmCols :", rdmCols)
	sub := subsample(m, ratioR, append(rdmCols, ycol))
	fmt.Print("sub matrix\n", mat.Formatted(sub), "\n")

	tree := decision.Fit(sub, 2, 2, 1)

	fmt.Println(tree)

	t.Fail()
}
