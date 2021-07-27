import numpy as np
from losses.loss import Loss


# Mean Absolute Error loss
# noinspection PyPep8Naming
class Loss_MeanAbsoluteError(Loss):  # L1 loss

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        # Calculate loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)

        # Return losses
        return sample_losses

    # Backward pass
    def backward(self, dvalues: np.ndarray, y_true: np.ndarray):
        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])

        # Calculate gradient
        self.dinputs = np.sign(y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs /= samples
