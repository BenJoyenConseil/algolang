package rf

import (
	"fmt"
	"rf/decision"
	"rf/ensemble"
	"rf/eval"
	"rf/io"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/tobgu/qframe/config/csv"
)

func TestFunctional_DecisionTree(t *testing.T) {

	types := map[string]string{"y": "float"}
	df := io.LoadCsv("./data/data_banknote_authentication.txt", csv.Headers([]string{"col_0", "col_1", "col_2", "col_3", "y"}), csv.Types(types))
	m := io.ToMatrix(df)

	model := decision.Fit(m, -1, 5, 10)
	preds := model.Predict(m)
	y, _ := df.FloatView("y")
	a := eval.Accuracy(y.Slice(), preds)
	fmt.Println(model)
	fmt.Println(a)

	assert.Greater(t, a, 97.0)
}

func BenchmarkFit_DecisionTree(b *testing.B) {
	for i := 0; i < b.N; i++ {

		types := map[string]string{"y": "float"}
		df := io.LoadCsv("./data/data_banknote_authentication.txt", csv.Headers([]string{"col_0", "col_1", "col_2", "col_3", "y"}), csv.Types(types))
		m := io.ToMatrix(df)
		_ = decision.Fit(m, -1, 5, 10)
	}
}

func TestFunctional_RandomForest(t *testing.T) {

	types := map[string]string{"y": "float"}
	df := io.LoadCsv("./data/data_banknote_authentication.txt", csv.Headers([]string{"col_0", "col_1", "col_2", "col_3", "y"}), csv.Types(types))
	m := io.ToMatrix(df)

	model := ensemble.Fit(m, -1, 5, 5, 10)
	preds := model.Predict(m)
	y, _ := df.FloatView("y")
	a := eval.Accuracy(y.Slice(), preds)

	fmt.Println(model)
	fmt.Println("Accuracy", a)

	assert.Greater(t, a, 90.0)
}

func BenchmarkFit_RandomForest(b *testing.B) {
	for i := 0; i < b.N; i++ {

		types := map[string]string{"y": "float"}
		df := io.LoadCsv("./data/data_banknote_authentication.txt", csv.Headers([]string{"col_0", "col_1", "col_2", "col_3", "y"}), csv.Types(types))
		m := io.ToMatrix(df)
		_ = ensemble.Fit(m, -1, 5, 5, 10)
	}
}
