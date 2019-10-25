package rf

import (
	"encoding/csv"
	"fmt"
	"io"
	"strconv"
	"strings"
)

/*
Serie is a slice of float64. It represents all values of the column
*/
type Serie []float64

/*
Dataset is float64 matrix, or a set of Serie/column data
*/
type Dataset []Serie

type DataFrame map[string]Serie

func NewDataFrame() DataFrame {
	return DataFrame(make(map[string]Serie))
}

func (d DataFrame) Add(s Serie) DataFrame {
	d[strconv.FormatInt(int64(len(d)), 10)] = s
	return d
}

func (d DataFrame) Drop(column string) DataFrame {
	new := NewDataFrame()
	for col, serie := range d {
		if col != column {
			new[col] = serie
		}
	}
	return new
}

func (d DataFrame) Size() int {
	for _, s := range d {
		return len(s)
	}
	return 0
}

func (d DataFrame) AddRow(row []float64) DataFrame {
	i := 0
	for col, serie := range d {
		fmt.Println(col)
		d[col] = append(serie, row[i])
		i++
	}
	return d
}

func (d DataFrame) ILoc(index int) (row []float64) {
	row = make([]float64, len(d))
	i := 0
	for _, serie := range d {
		row[i] = serie[index]
		i++
	}
	return row
}

func Concat(left, right DataFrame) DataFrame {
	df := NewDataFrame()
	if left.Size() != right.Size() {
		panic("Left and Right have not the same columns")
	}
	for col, serie := range left {
		df[col] = append(serie, right[col]...)
	}
	return df
}

func ReadCSV(r io.ReadSeeker) DataFrame {
	df := NewDataFrame()
	csvReader := csv.NewReader(r)

	// initialize column Series
	record, _ := csvReader.Read()
	cols := make(map[int]string)
	for i, field := range record {
		trim := strings.TrimSpace(field)
		// if no header
		if value, err := strconv.ParseFloat(trim, 64); err == nil {
			df = df.Add(Serie{value})
			cols[i] = strconv.FormatInt(int64(i), 10)
		} else /* if headers  */ {
			df[trim] = Serie{}
			cols[i] = trim
		}
	}
	for {
		record, err := csvReader.Read()
		if err == io.EOF {
			break
		}
		for i, field := range record {
			trim := strings.TrimSpace(field)
			if value, err := strconv.ParseFloat(trim, 64); err == nil {
				df[cols[i]] = append(df[cols[i]], value)
			}
		}
	}
	return df
}
