package rf

func Accuracy(actual, predicted []int) float32{
	var correct int = 0

	for i := 0; i < len(actual); i++ {

		if predicted[i] == actual[i] {
			correct ++
		}
	}
	return float32(correct) / float32(len(actual)) * 100.0
}

