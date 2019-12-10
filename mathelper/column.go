package mathelper

import (
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/mat"
)

// column is a user-defined column vector.
type Column []float64

// Dims, At and T minimally satisfy the mat.Matrix interface.
func (v Column) Dims() (r, c int)    { return len(v), 1 }
func (v Column) At(i, _ int) float64 { return v[i] }
func (v Column) T() mat.Matrix       { return Row(v) }

// AtVec, Len minimally satisfy the mat.Vector interface.
func (v Column) AtVec(i int) float64 { return v[i] }
func (v Column) Len() int            { return len(v) }

// RawVector allows fast path computation with the vector.
func (v Column) RawVector() blas64.Vector {
	return blas64.Vector{N: len(v), Data: v, Inc: 1}
}
