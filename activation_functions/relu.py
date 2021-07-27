import numpy as np


# ReLU activation
# noinspection PyPep8Naming
from activation_functions.activation_function import ActivationFunction


class Activation_ReLU(ActivationFunction):

    # Forward pass
    def forward(self, inputs: np.ndarray):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues: np.ndarray):
        # Since we need to modify original variable,
        # let's make a copy of values first
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs
