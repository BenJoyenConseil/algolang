package rf

func Accuracy(actual, predicted []float64) float64 {
	var correct int = 0

	for i := 0; i < len(actual); i++ {

		if predicted[i] == actual[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(actual)) * 100.0
}
