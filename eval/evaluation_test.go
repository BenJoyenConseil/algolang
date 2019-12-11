package eval

import (
	"rf/algo"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

type mockModel struct{}

func (mock *mockModel) Predict(rows *mat.Dense) (predictions []float64) {
	dR, _ := rows.Dims()
	return make([]float64, dR)
}
func (mock mockModel) IsFitted() bool                { return true }
func (mock mockModel) PredictRow(mat.Vector) float64 { return 0. }

func returnAlways0ModelFit(m *mat.Dense, yCol int, params map[string]int) algo.Model {
	return &mockModel{}
}

func TestCrossVal(t *testing.T) {
	// Given
	USELESS := -1.
	m := mat.NewDense(10, 2, []float64{
		USELESS, 1,
		USELESS, 1,
		USELESS, 1,
		USELESS, 1,
		USELESS, 1,
		USELESS, 1,
		USELESS, 1,
		USELESS, 0,
		USELESS, 0,
		USELESS, 0})

	// When
	scores := CrossVal(m, 1, 5, returnAlways0ModelFit, map[string]int{})

	// Then
	assert.Equal(t, scores[0], 0.0)
	assert.Equal(t, scores[1], 0.0)
	assert.Equal(t, scores[2], 0.0)
	assert.Equal(t, scores[3], 50.0)
	assert.Equal(t, scores[4], 100.0)

}

func TestTrainTestSplit(t *testing.T) {
	// Given
	m := mat.NewDense(10, 1, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})

	// When
	train, test := splitTrainTest(m, 5, 5)

	// Then
	assert.Equal(t, train.At(0, 0), 1.)
	assert.Equal(t, train.At(1, 0), 2.)
	assert.Equal(t, train.At(2, 0), 3.)
	assert.Equal(t, train.At(3, 0), 4.)
	assert.Equal(t, train.At(4, 0), 5.)
	assert.Equal(t, train.At(5, 0), 6.)
	assert.Equal(t, train.At(6, 0), 7.)
	assert.Equal(t, train.At(7, 0), 8.)
	assert.Equal(t, test.At(0, 0), 9.)
	assert.Equal(t, test.At(1, 0), 10.)

	// When
	train, test = splitTrainTest(m, 2, 5)

	// Then
	assert.Equal(t, train.At(0, 0), 1.)
	assert.Equal(t, train.At(1, 0), 2.)
	assert.Equal(t, train.At(2, 0), 5.)
	assert.Equal(t, train.At(3, 0), 6.)
	assert.Equal(t, train.At(4, 0), 7.)
	assert.Equal(t, train.At(5, 0), 8.)
	assert.Equal(t, train.At(6, 0), 9.)
	assert.Equal(t, train.At(7, 0), 10.)
	assert.Equal(t, test.At(0, 0), 3.)
	assert.Equal(t, test.At(1, 0), 4.)

	// Given
	m = mat.NewDense(11, 1, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11})

	// When
	train, test = splitTrainTest(m, 5, 5)

	// Then
	tdR, tdC := train.Dims()
	assert.Equal(t, 8, tdR)
	assert.Equal(t, 1, tdC)
	tdR, tdC = test.Dims()
	assert.Equal(t, 3, tdR)
	assert.Equal(t, 1, tdC)
	assert.Equal(t, train.At(0, 0), 1.)
	assert.Equal(t, train.At(1, 0), 2.)
	assert.Equal(t, train.At(2, 0), 3.)
	assert.Equal(t, train.At(3, 0), 4.)
	assert.Equal(t, train.At(4, 0), 5.)
	assert.Equal(t, train.At(5, 0), 6.)
	assert.Equal(t, train.At(6, 0), 7.)
	assert.Equal(t, train.At(7, 0), 8.)
	assert.Equal(t, test.At(0, 0), 9.)
	assert.Equal(t, test.At(1, 0), 10.)
	assert.Equal(t, test.At(2, 0), 11.)
}
