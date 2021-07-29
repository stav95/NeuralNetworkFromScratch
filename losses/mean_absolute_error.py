import numpy as np
from losses.loss import Loss


# noinspection PyPep8Naming
class Loss_MeanAbsoluteError(Loss):  # L1 loss
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        return sample_losses

    def backward(self, dvalues: np.ndarray, y_true: np.ndarray):
        # TODO: dvalues_clipped = super().backward(dvalues=dvalues, y_true=y_true)
        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])

        # Calculate gradient
        self.dinputs = np.sign(y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs /= samples
