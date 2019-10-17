package rf

import (
	"fmt"
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
		s := fmt.Sprintf("Accuracy %v != 50", r)
		t.Error(s)
	}
	if r2 != 100 {
		s := fmt.Sprintf("Accuracy %v != 100", r2)
		t.Error(s)
	}
	if r3 != 0 {
		s := fmt.Sprintf("Accuracy %v != 0", r3)
		t.Error(s)
	}
}
