package rf

import (
	"fmt"
)

type Dataset [][]float64

type Tree struct {
	Left  *Tree
	Value float64
	Right *Tree
}

func giniIndex(left, right Dataset, classes []float64) (gini float64) {
	nSamples := len(left) + len(right)

	gini = 0.0
	rowGroups := []Dataset{left, right}
	for _, rows := range rowGroups {
		gSize := float64(len(rows))
		if gSize == 0.0 {
			continue
		}
		score := 0.0
		for _, c := range classes {
			classHit := 0.0
			for _, r := range rows {
				if y := r[len(r)-1]; y == c {
					classHit++
				}
			}
			p := classHit / gSize
			score += p * p
		}
		gini += (1.0 - score) * (gSize / float64(nSamples))
	}
	return
}

func split(dataset Dataset, idFeature int, threshold float64) (left, right Dataset) {

	for _, row := range dataset {
		if row[idFeature] < threshold {
			left = append(left, row)
		} else {
			right = append(right, row)
		}
	}
	return
}

func bestSplit(dataset Dataset) (idFeature int, threshold float64, score float64, left, right Dataset) {
	idFeature, threshold, score = 999, 999.0, 999.0
	classesMap := map[float64]int{}
	for _, row := range dataset {
		classesMap[row[len(row)-1]] = 1
	}
	classes := []float64{}
	for k := range classesMap {
		classes = append(classes, k)
	}

	for idCol := 0; idCol < len(dataset[0])-1; idCol++ {
		for _, row := range dataset {
			l, r := split(dataset, idCol, row[idCol])
			gini := giniIndex(l, r, classes)
			fmt.Println("X", idCol, "<", row[idCol], "Gini :", gini)
			if gini < score {
				idFeature, threshold, score, left, right = idCol, row[idCol], gini, l, r
			}
			if score == 0.0 {
				return
			}
		}
	}
	return
}

func unique(tab []float64) []float64 {
	m := map[float64]bool{}
	for _, c := range tab {
		m[c] = true
	}
	uniqueVal := []float64{}
	for k := range m {
		uniqueVal = append(uniqueVal, k)
	}

	return uniqueVal
}

func vote(predictions []float64) float64 {
	mapPred := make(map[float64]int)
	for _, p := range predictions {
		if val, ok := mapPred[p]; ok {
			mapPred[p] = val + 1
		} else {
			mapPred[p] = 1
		}
		fmt.Println(mapPred[p])
	}
	maxV := mapPred[predictions[0]]
	maxIndex := predictions[0]
	for k, v := range mapPred {
		fmt.Println(v)
		if v > maxV {
			maxV = v
			maxIndex = k
		}
	}
	return maxIndex
}
