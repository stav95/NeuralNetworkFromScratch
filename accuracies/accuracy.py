import numpy as np
import abc


# noinspection PyPep8Naming
class Accuracy(metaclass=abc.ABCMeta):
    def __init__(self):
        self.accumulated_accuracy = 0
        self.accumulated_count = 0

    @abc.abstractmethod
    def compare(self, predictions: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass

    def calculate(self, predictions: np.ndarray, y: np.ndarray) -> float:
        # Get comparison results
        comparisons = self.compare(predictions=predictions, y=y)

        # Calculate an accuracy from array like [True=1, False=0, True=1,...]
        accuracy = comparisons.mean()

        # Add accumulated sum of matching values and sample count
        self.accumulated_accuracy += np.sum(comparisons)
        self.accumulated_count += comparisons.shape[0]

        return accuracy

    def calculate_accumulated_accuracy(self) -> float:
        accuracy = self.accumulated_accuracy / self.accumulated_count
        return accuracy

    def new_pass(self):
        self.accumulated_accuracy = 0
        self.accumulated_count = 0
