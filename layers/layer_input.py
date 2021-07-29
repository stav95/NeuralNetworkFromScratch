import numpy as np
from layers.layer import Layer


# noinspection PyPep8Naming
class Layer_Input(Layer):
    def forward(self, inputs: np.ndarray):
        self.output = inputs

    def backward(self, dvalues: np.ndarray):
        pass
