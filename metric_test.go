package rf

import (
	"testing"
)

func TestAccuracy(t *testing.T) {
	// Given
	actual := []int{1, 0}
	predicted := []int{ /* first test */ 1, 1 /* second test */, 1, 0 /* third test */, 0, 1}

	// When
	r := Accuracy(actual, predicted[:2])
	r2 := Accuracy(actual, predicted[2:4])
	r3 := Accuracy(actual, predicted[4:])

	// Then
	if r != 50 {
		t.Error("Should be Accuracy 50 !=", r)
	}
	if r2 != 100 {
		t.Error("Should be Accuracy 100 !=", r2)
	}
	if r3 != 0 {
		t.Error("Should be Accuracy 0 !=", r3)
	}
}
