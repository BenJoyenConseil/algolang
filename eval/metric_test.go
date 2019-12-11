package eval

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestAccuracy(t *testing.T) {
	// Given
	actual := []float64{1, 0}
	predicted := []float64{ /* first test */ 1, 1 /* second test */, 1, 0 /* third test */, 0, 1}

	// When
	r := Accuracy(actual, predicted[:2])
	r2 := Accuracy(actual, predicted[2:4])
	r3 := Accuracy(actual, predicted[4:])

	// Then
	assert.Equal(t, 50., r)
	assert.Equal(t, 100., r2)
	assert.Equal(t, 0., r3)
}
