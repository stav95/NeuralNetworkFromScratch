import numpy as np
from layers.layer import Layer


# noinspection PyPep8Naming
class Layer_Dropout(Layer):
    def __init__(self, rate: float):
        super().__init__()

        # Store rate, we invert it as for example for dropout
        # of 0.1 we need success rate of 0.9
        self.rate = 1 - rate

    # Forward pass
    def forward(self, inputs: np.ndarray):
        self.inputs = inputs

        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask

    def backward(self, dvalues: np.ndarray):
        self.dinputs = dvalues * self.binary_mask
