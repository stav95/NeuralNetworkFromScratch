import abc
import numpy as np


# Common accuracies class
# noinspection PyPep8Naming
class Accuracy(metaclass=abc.ABCMeta):
    def __init__(self):
        self.accumulated_accuracy = 0
        self.accumulated_count = 0

    @abc.abstractmethod
    def compare(self, predictions: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass

    # Calculates an accuracies
    # given predictions and ground truth values
    def calculate(self, predictions: np.ndarray, y: np.ndarray) -> float:
        # Get comparison results
        comparisons = self.compare(predictions=predictions, y=y)

        # Calculate an accuracies
        accuracy = comparisons.mean()

        # Add accumulated sum of matching values and sample count
        self.accumulated_accuracy += np.sum(comparisons)
        self.accumulated_count += comparisons.shape[0]

        # Return accuracies
        return accuracy

    # Calculates accumulated accuracies
    def calculate_accumulated_accuracy(self) -> float:
        # Calculate an accuracies
        accuracy = self.accumulated_accuracy / self.accumulated_count

        # Return the data and regularization losses
        return accuracy

    # Reset variables for accumulated accuracies
    def new_pass(self):
        self.accumulated_accuracy = 0
        self.accumulated_count = 0
