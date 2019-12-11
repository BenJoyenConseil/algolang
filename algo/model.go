package algo

import "gonum.org/v1/gonum/mat"

// Model is an abstraction of estimators provided in the ensemble package
type Model interface {
	Predict(m *mat.Dense) (predictions []float64)
	IsFitted() bool
	PredictRow(v mat.Vector) float64
}
