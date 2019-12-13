# Algolang
Machine Learning algorithms : 
Decision Tree, Random Forest implementations


## Algorithms & Usage
"This section applies the CART algorithm to the Bank Note dataset."

### Load CSV file into master Matrix
```go
types := map[string]string{"y": "float"}
df := io.LoadCsv("./testdata/data_banknote_authentication.txt", csv.Headers([]string{"col_0", "col_1", "col_2", "col_3", "y"}), csv.Types(types))
m := io.ToMatrix(df)
```
### The cross validation with the  accuracy score on DecisionTree
```go
scores := eval.CrossVal(m, 4, 5, decision.Fit, map[string]int{"maxDepth": 5, "minSize": 10})
fmt.Println("Decision Tree", scores)
```
Output :
```c
Decision Tree [95.25547445255475 98.17518248175182 96.71532846715328 90.51094890510949 98.91304347826086]
```
### The cross validation with the accuracy score on RandomForest
```go
scores = eval.CrossVal(m, 4, 5, ensemble.Fit, map[string]int{"n_estimator": 5, "maxDepth": 5, "minSize": 10})
fmt.Println("RandoForest", scores)
```
Output :
```c
RandoForest [90.14598540145985 89.78102189781022 96.71532846715328 92.33576642335767 93.11594202898551]
```

## Packages

* [io /](./io) : it has `ReadCSV` that returns a `QFrame` (like pandas.DataFrame for golang). `ToMatrix` takes a `QFrame` and return a `gonum.mat.Dense` object

* [mathelper /](./mathelper) : matrix helpers like `[]float64` to `gonum.mat.Vector` convertion (into a `Row` or `Column` object). There is a `Mode` (statistic) function taking a `gonum.mat.Vector`

* [eval /](./eval) : has `Accuracy` score function in `metric.go` and expose `CrossVal` that takes an algo `Fit` function and return an array of the resultted accuracy scores for many folds

* [algo /](./algo)
    * `model.go` : defines the `Model` interface which has `Predict` contract.
    * [decision /](./algo/decision) : DecisionTree is exposed by this package, using CART and the gini function.
    * [ensemble /](./algo/ensemble) : RandomForest algorithm is exposed by this package. It uses Boostraping and Bagging of DecisionTrees.
