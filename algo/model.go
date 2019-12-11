package algo

import "gonum.org/v1/gonum/mat"

// Model is an abstraction of estimators provided in the ensemble package
type Model interface {
	Predictor
	IsFitted() bool
	PredictRow(v mat.Vector) float64
}

type Predictor interface {
	Predict(m *mat.Dense) (predictions []float64)
}
