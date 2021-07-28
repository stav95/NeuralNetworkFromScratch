import numpy as np


# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
# noinspection PyPep8Naming
class Activation_Softmax_Loss_CategoricalCrossentropy:

    # Backward pass
    def backward(self, dvalues: np.ndarray, y_true: np.ndarray):
        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[np.arange(dvalues.shape[0]), y_true] -= 1
        # Normalize gradient
        self.dinputs /= dvalues.shape[0]
