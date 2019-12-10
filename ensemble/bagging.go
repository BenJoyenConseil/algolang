package ensemble

import (
	"math"
	"math/rand"
	"rf/ensemble/decision"
	"rf/mathelper"
	"sort"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

/*
RandoForest is a bagging algorithm based on decision Trees
*/
type RandomForest struct {
	estimators []*decision.Tree
	Score      float64
	// feMapping store the mapping between subtrees that learn only on a subset of the feature and
	// the general Matrix passed to the Fit function.
	// The feMapping[treePointer][subMatrixFeatureId] = generalMatrixIdFeature
	feMapping map[*decision.Tree][]int
}

/*
Fit builds decision trees on subsamples of the matrix X using the sqare root of nFeatures
*/
func Fit(m *mat.Dense, yCol int, nEstimators, maxDepth, minSize int, seed int64) *RandomForest {
	feCols := extractFeatures(m, yCol)
	rf := &RandomForest{
		feMapping: make(map[*decision.Tree][]int),
	}
	ratioR := 1.0
	ratioC := 1 - sqrtRatio(len(feCols))

	for estimator := 0; estimator < nEstimators; estimator++ {
		subCols := randomSubColumns(feCols, ratioC, seed)
		subM := subsample(m, ratioR, append(subCols, yCol), seed)
		t := decision.Fit(subM, -1, maxDepth, minSize)
		rf.estimators = append(rf.estimators, t)
		rf.feMapping[t] = subCols
	}
	return rf
}

// Predict returns an array of predictions for each row in the Matrix
func (rf *RandomForest) Predict(m *mat.Dense) (predictions []float64) {
	dR, _ := m.Dims()
	predictions = make([]float64, dR)
	for i := 0; i < dR; i++ {
		predictions[i] = rf.PredictRow(m.RowView(i))
	}
	return predictions
}

// PredictRow returns the most frequent predictions accross all estimators predictions
func (rf *RandomForest) PredictRow(row mat.Vector) float64 {
	var predictions mathelper.Row = make([]float64, len(rf.estimators))
	for i, estimator := range rf.estimators {
		features := rf.feMapping[estimator]
		projectedRow := make([]float64, len(features))
		for i, f := range features {
			projectedRow[i] = row.AtVec(f)
		}
		predictions[i] = estimator.PredictRow(mathelper.Row(projectedRow))
	}
	mode, _ := stat.Mode(predictions, nil)
	return mode
}

// IsFitted returns False if NFeatures is <= 0 or Score < 0 or treeBag length is < 0
func (rf *RandomForest) IsFitted() bool {
	if len(rf.estimators) >= 0 && rf.Score >= 0 {
		return true
	}
	return false
}

func extractFeatures(m mat.Matrix, yCol int) []int {
	feCols := []int{}
	_, dC := m.Dims()
	for c := 0; c < dC; c++ {
		if c != yCol {
			feCols = append(feCols, c)
		}
	}
	return feCols
}

func subsample(m *mat.Dense, ratio float64, columns []int, seed int64) (samples mat.Matrix) {
	r, _ := m.Dims()
	nRow := int(float64(r) * ratio)
	sub := mat.NewDense(nRow, len(columns), nil)
	s := rand.NewSource(seed)
	random := rand.New(s)
	for i := 0; i < nRow; i++ {
		id := random.Intn(r)
		row := m.RawRowView(id)
		for j, cid := range columns {
			sub.Set(i, j, row[cid])
		}
	}
	return sub
}

func randomSubColumns(columns []int, ratio float64, seed int64) []int {
	s := rand.NewSource(seed)
	random := rand.New(s)
	n := int(ratio * float64(len(columns)))
	indexes := make(map[int]bool, n)
	cols := make([]int, 0, n)

	for len(indexes) < n {
		indexes[random.Intn(n)] = true
	}
	for k := range indexes {
		cols = append(cols, k)
	}
	sort.Ints(cols)
	return cols
}

func sqrtRatio(n int) float64 {
	sqrt := math.Sqrt(float64(n))
	return math.Round(1/sqrt*10) / 10
}
