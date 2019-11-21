package io

import (
	"os"

	"github.com/tobgu/qframe"
	"github.com/tobgu/qframe/config/csv"
	"gonum.org/v1/gonum/mat"
)

func LoadCsv(filename string) qframe.QFrame {
	csvFile, err := os.Open(filename)
	if err != nil {
		panic(err)
	}
	types := map[string]string{"y": "float"}
	return qframe.ReadCSV(csvFile, csv.Headers([]string{"col_0", "col_1", "col_2", "col_3", "y"}), csv.Types(types))
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
