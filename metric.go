package rf

/*
Accuracy counts the number of predictions equal to the actual class and returns it as percentage
*/
func Accuracy(actual, predicted Serie) float64 {
	var correct int = 0

	for i := 0; i < len(actual); i++ {

		if predicted[i] == actual[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(actual)) * 100.0
}
