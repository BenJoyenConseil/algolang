# algolang
Machine Learning algorithms : 
Decision Tree, Random Forest implementations


## Algorithms & Usage
"This section applies the CART algorithm to the Bank Note dataset."

The ```tree``` module provides Fit constructor that returns a DecisionTree

	model := Fit(df, 5, 10)
	preds := model.Predict(df.Drop("y"))
	
The ```eval``` module includes tools to evaluate your model

	a := Accuracy(df["y"], preds)
	fmt.Println(a)


### a DataFrame little interface

DataFrame interface is an array of columns (Serie), and Series are arrays of Float64

	type Serie []float64
	type DataFrame map[string]Serie


Loading a CSV file. If there is no header, columns have its index number as the key

	df := loadCsv("./data_banknote_authentication.txt")

Creates a new Column named ```y``` with values Serie of the column identified by ```4```

	df["y"] = df["4"]

Drop the ```4``` column
	
	df = df.Drop("4")	
	
You can create a new DataFrame like this

	df2 := DataFrame{
		"x0":   {2.771244718, 1.728571309, 3.678319846, 3.961043357, 2.999208922, 7.497545867, 9.00220326, 7.444542326, 10.12493903, 6.642287351},
		"x1":   {1.784783929, 1.169761413, 2.81281357, 2.61995032, 2.209014212, 3.162953546, 3.339047188, 0.476683375, 3.234550982, 3.319983761},
		"yeah": {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0},
	}
	
Concatenation of 2 DataFrames

	Concat(df, df2)
	
Accessing a row by its index return an array of Float64
	
	df.ILoc(0)
	
	
	df["0"]
	

**Serie** only supportes Float64 actually. Futur => [QFrame](https://github.com/tobgu/qframe) integration
