package io

import (
	"os"

	"github.com/tobgu/qframe"
	"github.com/tobgu/qframe/config/csv"
	"gonum.org/v1/gonum/mat"
)

// LoadCsv loads a csv into a QFrame structure
func LoadCsv(filename string, headers ...csv.ConfigFunc) qframe.QFrame {
	csvFile, err := os.Open(filename)
	if err != nil {
		panic(err)
	}
	return qframe.ReadCSV(csvFile, headers...)
}

func ToMatrix(df qframe.QFrame) *mat.Dense {
	data := []float64{}
	for i := 0; i < df.Len(); i++ {
		for _, c := range df.ColumnNames() {
			v, _ := df.FloatView(c)
			data = append(data, v.ItemAt(i))
		}
	}
	return mat.NewDense(df.Len(), len(df.ColumnNames()), data)
}
