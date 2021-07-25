import numpy as np


# Common accuracies class
# noinspection PyPep8Naming
class Accuracy:

    # Calculates an accuracies
    # given predictions and ground truth values
    def calculate(self, predictions: np.ndarray, y: np.ndarray) -> float:
        # Get comparison results
        comparisons = self.compare(predictions, y)

        # Calculate an accuracies
        accuracy = comparisons.mean()

        # Add accumulated sum of matching values and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += comparisons.shape[0]

        # Return accuracies
        return accuracy

    # Calculates accumulated accuracies
    def calculate_accumulated(self) -> float:
        # Calculate an accuracies
        accuracy = self.accumulated_sum / self.accumulated_count

        # Return the data and regularization losses
        return accuracy

    # Reset variables for accumulated accuracies
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
