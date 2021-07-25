import numpy as np

# Linear activation
# noinspection PyPep8Naming
class Activation_Linear:

    # Forward pass
    def forward(self, inputs: np.ndarray, training: bool):
        # Just remember values
        self.inputs = inputs
        self.output = inputs

    # Backward pass
    def backward(self, dvalues):
        # derivative is 1, 1 * dvalues = dvalues - the chain rule
        self.dinputs = dvalues.copy()

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs
