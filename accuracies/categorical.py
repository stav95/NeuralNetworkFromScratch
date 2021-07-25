import numpy as np
from accuracies.accuracy import Accuracy


# Accuracy calculation for classification model
# noinspection PyPep8Naming
class Accuracy_Categorical(Accuracy):
    def __init__(self, *, binary: bool = False):
        # Binary mode?
        self.binary = binary

    # No initialization is needed
    def init(self, y: np.ndarray):
        pass

    # Compares predictions to the ground truth values
    def compare(self, predictions: np.ndarray, y: np.ndarray) -> np.ndarray:
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y
