import numpy as np
from losses.loss import Loss


# noinspection PyPep8Naming
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        correct_confidences = None

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values - only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[np.arange(y_pred.shape[0]), y_true]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(np.multiply(y_pred_clipped, y_true), axis=1)

        return -np.log(correct_confidences)  # Losses

    def backward(self, dvalues: np.ndarray, y_true: np.ndarray):
        # Clip data to prevent division by 0
        dvalues_clipped = np.clip(dvalues, 1e-7, 1)

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(dvalues.shape[1])[y_true]

        # Calculate gradient
        self.dinputs = -np.divide(y_true, dvalues_clipped)

        # Normalize gradient
        self.dinputs = np.divide(self.dinputs, dvalues.shape[0])
