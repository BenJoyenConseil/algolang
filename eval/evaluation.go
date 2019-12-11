package eval

import (
	"rf/algo"

	"gonum.org/v1/gonum/mat"
)

// CrossVal launches n-times train-test-split, fits on train, and predict on test.
// It returns accuracy on each fold-iteration
func CrossVal(m *mat.Dense, yCol int, nFold int, f func(*mat.Dense, int, map[string]int) algo.Model, params map[string]int) (scores []float64) {
	for n := 1; n <= nFold; n++ {
		train, test := splitTrainTest(m, n, nFold)
		model := f(train, yCol, params)
		y := mat.Col(nil, yCol, test)
		scores = append(scores, Accuracy(y, model.Predict(test)))
	}
	return
}

func splitTrainTest(m *mat.Dense, n, nFold int) (train, test *mat.Dense) {
	dR, dC := m.Dims()

	for i := 0; i < nFold; i++ {
		xStart := i * int(dR/nFold)
		xEnd := (i + 1) * int(dR/nFold)
		if (dR - xEnd) < int(dR/nFold) {
			xEnd = dR
		}
		fold := m.Slice(xStart, xEnd, 0, dC).(*mat.Dense)
		if i != n-1 {

			if train == nil {
				train = fold
			} else {
				concat := &mat.Dense{}
				concat.Stack(train, fold)
				train = concat
			}
		} else {
			test = fold
		}
	}
	return
}
