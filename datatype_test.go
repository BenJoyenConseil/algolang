package rf

import (
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestAdd(t *testing.T) {
	// Given
	df := NewDataFrame()

	// When
	r := df.Add(Serie{}).Add(Serie{})

	// Then
	assert.NotNil(t, r["0"])
	assert.NotNil(t, r["1"])
}

func TestDrop(t *testing.T) {
	// Given
	df := NewDataFrame()
	df["col_delete_me"] = Serie{}

	// When
	df = df.Drop("col_delete_me")

	// Then
	assert.Nil(t, df["col_delete_me"])
}

func TestReadCSV_Header(t *testing.T) {
	// Given
	csvStr := `
	Field1, Field2, y
	1.0, 2.0, 3.0,
	`
	reader := strings.NewReader(csvStr)

	// When
	df := ReadCSV(reader)

	// Then
	assert.Equal(t, 1.0, df["Field1"][0])
	assert.Equal(t, 2.0, df["Field2"][0])
	assert.Equal(t, 3.0, df["y"][0])
}

func TestReadCSV_NoHeader(t *testing.T) {
	// Given
	csvStr := "1.0, 2.0, 3.0"
	reader := strings.NewReader(csvStr)

	// When
	df := ReadCSV(reader)

	// Then
	assert.Equal(t, 1.0, df["0"][0])
	assert.Equal(t, 2.0, df["1"][0])
	assert.Equal(t, 3.0, df["2"][0])
}

func TestSize(t *testing.T) {
	// Given
	df := NewDataFrame()
	df["0"] = Serie{1, 2, 3}

	// When
	r := df.Size()

	// Then
	assert.Equal(t, 3, r)
}

func TestAddRow(t *testing.T) {
	// Given
	df := NewDataFrame()
	df["x0"] = Serie{1.0}
	df["x1"] = Serie{1.0}

	// When
	df.AddRow([]float64{2.0, 2.0})

	// Then
	assert.ElementsMatch(t, df["x0"], []float64{1.0, 2.0})
	assert.ElementsMatch(t, df["x1"], []float64{1.0, 2.0})
}

func TestILoc(t *testing.T) {
	// Given
	df := NewDataFrame()
	df["x0"] = Serie{1.0, 2.0}
	df["x1"] = Serie{3.0, 4.0}

	// When
	r := df.ILoc(0)
	r2 := df.ILoc(1)

	// Then
	assert.ElementsMatch(t, r, []float64{1.0, 3.0})
	assert.ElementsMatch(t, r2, []float64{2.0, 4.0})
}

func TestSortKeys(t *testing.T) {
	// Given
	df := NewDataFrame()
	df["x0"] = Serie{1.0, 2.0}
	df["x1"] = Serie{3.0, 4.0}
	df["x2"] = Serie{3.0, 4.0}

	// When
	i := 0
	for _, col := range df.SortKeys() {
		assert.Equal(t, fmt.Sprintf("x%v", i), col)
		i++
	}

	// Then
}

func TestDataFrame_APIusage(t *testing.T) {
	df := NewDataFrame()

	df["X1"] = Serie{1.0, 0.2, 9.092}
	df["X2"] = Serie{2.0, 2.2, 11.092}
	df["y"] = Serie{1.0, 1.0, 0.0}
	df["preds"] = Serie{1.0, 0.0, 1.0}
	for _, row := range df["y"] {
		fmt.Println(row)
	}
	for i, col := range df {
		fmt.Println(i, col)
	}

}
