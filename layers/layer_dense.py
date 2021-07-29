import numpy as np
from typing import Tuple
from layers.layer import Layer


# noinspection PyPep8Naming
class Layer_Dense(Layer):
    def __init__(self,
                 n_inputs: int,
                 n_neurons: int,
                 weight_regularizer_l1: float = 0,
                 weight_regularizer_l2: float = 0,
                 bias_regularizer_l1: float = 0,
                 bias_regularizer_l2: float = 0):
        super().__init__()

        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def get_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.weights, self.biases

    def set_parameters(self, weights: np.ndarray, biases: np.ndarray):
        self.weights = weights
        self.biases = biases

    def forward(self, inputs: np.ndarray):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues: np.ndarray):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
