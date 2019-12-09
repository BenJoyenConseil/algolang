package ensemble

import (
	"math"
	"math/rand"
	"rf/ensemble/decision"
	"sort"

	"gonum.org/v1/gonum/mat"
)

/*
RandoForest is a bagging algorithm based on decision Trees
*/
type RandomForest struct {
	treeBag []*decision.Tree
	Score   float64
	// FeatureMap store the mapping between subtrees that learn only on a subset of the feature and
	// the general Matrix passed to the Fit function.
	// The featureMap[treePointer][subMatrixFeatureId] = generalMatrixIdFeature
	featureMap map[*decision.Tree]map[int]int
}

/*
Fit builds decision trees on subsamples of the matrix X using the sqare root of nFeatures
*/
func Fit(m *mat.Dense, yCol int, nEstimators, maxDepth, minSize int, seed int64) *RandomForest {
	feCols := extractFeatures(m, yCol)
	rf := &RandomForest{
		featureMap: make(map[*decision.Tree]map[int]int),
	}
	ratioR := 1.0
	ratioC := 1 - sqrtRatio(len(feCols))

	for estimator := 0; estimator < nEstimators; estimator++ {
		subCols := randomSubColumns(feCols, ratioC, seed)
		subM := subsample(m, ratioR, append(subCols, yCol), seed)
		t := decision.Fit(subM, -1, maxDepth, minSize)
		rf.treeBag = append(rf.treeBag, t)
	}
	return rf
}

func (rf *RandomForest) Predict(m *mat.Dense) (predictions []float64) {
	return nil
}

func (rf *RandomForest) PredictRow(row mat.Vector) float64 {
	return 0
}

// IsFitted returns False if NFeatures is <= 0 or Score < 0 or treeBag length is < 0
func (rf *RandomForest) IsFitted() bool {
	if len(rf.treeBag) >= 0 && rf.Score >= 0 {
		return true
	}
	return false
}

func extractFeatures(m mat.Matrix, yCol int) []int {
	feCols := []int{}
	_, nC := m.Dims()
	for c := 0; c < nC; c++ {
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
