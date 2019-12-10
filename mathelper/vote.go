package mathelper

import "gonum.org/v1/gonum/mat"

func Mode(v mat.Vector) float64 {
	counter := make(map[float64]int)
	l := v.Len()
	for i := 0; i < l; i++ {
		counter[v.AtVec(i)]++
	}
	max := counter[v.AtVec(0)]
	maxOccurence := v.AtVec(0)
	for k, v := range counter {
		if v > max {
			max = v
			maxOccurence = k
		}
	}
	return maxOccurence
}
