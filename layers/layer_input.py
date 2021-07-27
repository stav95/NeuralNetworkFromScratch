# Input "layer"
# noinspection PyPep8Naming
import numpy as np

from layers.layer import Layer


class Layer_Input(Layer):

    def forward(self, inputs: np.ndarray):
        self.output = inputs

    def backward(self, dvalues: np.ndarray):
        pass
