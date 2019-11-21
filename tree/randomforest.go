package tree

import "gonum.org/v1/gonum/mat"

type Forest struct {
	trees *Tree
}

func Fit(m mat.Matrix, forestSize int, numSampleSplit int) *Forest {
	return nil
}

func (rf *Forest) Predict(y mat.Vector) []float64 {
	return nil
}
