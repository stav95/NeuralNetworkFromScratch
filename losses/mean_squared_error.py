import numpy as np
from losses.loss import Loss


# Mean Squared Error loss
# noinspection PyPep8Naming
class Loss_MeanSquaredError(Loss):  # L2 loss

    # Forward pass
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        # Calculate loss
        sample_losses = np.mean(np.power(y_true - y_pred, 2), axis=-1)

        # Return losses
        return sample_losses

    # Backward pass
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
