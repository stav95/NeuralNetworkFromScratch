import numpy as np
from activation_functions.activation_function import ActivationFunction


# noinspection PyPep8Naming
class Activation_Sigmoid(ActivationFunction):
    def forward(self, inputs: np.ndarray):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues: np.ndarray):
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs: np.ndarray) -> np.ndarray:
        return (outputs > 0.5) * 1
