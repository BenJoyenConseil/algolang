package mathelper

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestMostCurrentValue(t *testing.T) {
	// Given
	predictions := mat.NewVecDense(5, []float64{0.3, 0.4, 0.9, 0.5, 0.3})
	predictions2 := Row{0.4, 0.4, 0.9, 0.9, 0.9}
	predictions3 := Row{0.1, 0.4, 0.8}

	// When
	r := MostCurrentValue(predictions)
	r2 := MostCurrentValue(predictions2)
	r3 := MostCurrentValue(predictions3)

	// Then
	assert.Equal(t, 0.3, r)
	assert.Equal(t, 0.9, r2)
	assert.Equal(t, 0.1, r3)
}
