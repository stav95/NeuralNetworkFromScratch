import numpy as np
from accuracies.accuracy import Accuracy


# Accuracy calculation for classification model
# noinspection PyPep8Naming
class Accuracy_Categorical(Accuracy):
    def __init__(self, binary: bool = False):
        self.binary = binary

    # Compares predictions to the ground truth values
    def compare(self, predictions: np.ndarray, y: np.ndarray) -> np.ndarray:
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y
