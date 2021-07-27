import numpy as np


# Sigmoid activation
# noinspection PyPep8Naming
from activation_functions.activation_function import ActivationFunction


class Activation_Sigmoid(ActivationFunction):

    # Forward pass
    def forward(self, inputs: np.ndarray):
        # Save input and calculate/save output
        # of the sigmoid function
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    # Backward pass
    def backward(self, dvalues: np.ndarray):
        # Derivative - calculates from output of the sigmoid function
        self.dinputs = dvalues * (1 - self.output) * self.output

    # Calculate predictions for outputs
    def predictions(self, outputs: np.ndarray):
        a = (outputs > 0.5) * 1
        return (outputs > 0.5) * 1
