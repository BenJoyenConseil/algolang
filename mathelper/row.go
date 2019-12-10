package mathelper

import (
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/mat"
)

// row is a user-defined row vector.
type Row []float64

// Dims, At and T minimally satisfy the mat.Matrix interface.
func (v Row) Dims() (r, c int)    { return 1, len(v) }
func (v Row) At(_, j int) float64 { return v[j] }
func (v Row) T() mat.Matrix       { return Column(v) }

// AtVec, Len minimally satisfy the mat.Vector interface.
func (v Row) AtVec(j int) float64 { return v[j] }
func (v Row) Len() int            { return len(v) }

// RawVector allows fast path computation with the vector.
func (v Row) RawVector() blas64.Vector {
	return blas64.Vector{N: len(v), Data: v, Inc: 1}
}
