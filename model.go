package rf

import "gonum.org/v1/gonum/mat"

// Model is an abstraction of estimators provided in the ensemble package
type Model interface {
	IsFitted() bool
	Predict(m *mat.Dense) (predictions []float64)
	PredictRow(v mat.Vector) float64
}
