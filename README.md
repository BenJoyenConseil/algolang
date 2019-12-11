# algolang
Machine Learning algorithms : 
Decision Tree, Random Forest implementations


## Algorithms & Usage
"This section applies the CART algorithm to the Bank Note dataset."

```ruby	
	types := map[string]string{"y": "float"}
	df := io.LoadCsv("./data/data_banknote_authentication.txt", csv.Headers([]string{"col_0", "col_1", "col_2", "col_3", "y"}), csv.Types(types))
	m := io.ToMatrix(df)

	scores := eval.CrossVal(m, 4, 5, decision.Fit, map[string]int{"maxDepth": 5, "minSize": 10})
	fmt.Println("Decision Tree", scores)

	scores = eval.CrossVal(m, 4, 5, ensemble.Fit, map[string]int{"n_estimator": 5, "maxDepth": 5, "minSize": 10})
	fmt.Println("RandoForest", scores)
```

Decision Tree [95.25547445255475 98.17518248175182 96.71532846715328 90.51094890510949 98.91304347826086]
RandoForest [90.14598540145985 89.78102189781022 96.71532846715328 92.33576642335767 93.11594202898551]

**Serie** only supportes Float64 actually. Futur => [QFrame](https://github.com/tobgu/qframe) integration
