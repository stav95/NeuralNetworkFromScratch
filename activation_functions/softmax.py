import numpy as np
from activation_functions.activation_function import ActivationFunction


# noinspection PyPep8Naming
class Activation_Softmax(ActivationFunction):
    def forward(self, inputs: np.ndarray):
        self.inputs = inputs

        # Get unnormalized probabilities
        # Substracting the max because we don't want to exp(very high number) --> inf --> error
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues: np.ndarray):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)

            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    def predictions(self, outputs: np.ndarray) -> np.ndarray:
        return np.argmax(outputs, axis=1)
