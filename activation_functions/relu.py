import numpy as np
from activation_functions.activation_function import ActivationFunction


# noinspection PyPep8Naming
class Activation_ReLU(ActivationFunction):
    def forward(self, inputs: np.ndarray):
        self.inputs = inputs

        self.output = np.maximum(0, inputs)

    def backward(self, dvalues: np.ndarray):
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs: np.ndarray) -> np.ndarray:
        return outputs
