# algolang
Machine Learning algorithms : 
Decision Tree, Random Forest implementations


## Usage
"This section applies the CART algorithm to the Bank Note dataset."

```
// dataset is a 2d slice [][]float64
dataset := loadCsv("./data_banknote_authentication.txt")
model := Fit(dataset[:], 5, 10)

y, preds := []float64{}, []float64{}
for _, row := range dataset[splitTrainSize:] {
	// make prediction
	pred := model.Predict(row[:len(row)-1])
	row = append(row, pred)
	y = append(y, row[len(row)-2])
	preds = append(preds, row[len(row)-1])
}
  
a := Accuracy(y, preds)
fmt.Println("Accuracy : ", a)
```
