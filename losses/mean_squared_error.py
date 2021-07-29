import numpy as np
from losses.loss import Loss


# noinspection PyPep8Naming
class Loss_MeanSquaredError(Loss):  # L2 loss
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        sample_losses = np.mean(np.power(y_true - y_pred, 2), axis=-1)
        return sample_losses

    def backward(self, dvalues: np.ndarray, y_true: np.ndarray):
        # TODO: dvalues_clipped = super().backward(dvalues=dvalues, y_true=y_true)
        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])

        # Gradient on values
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs /= samples
