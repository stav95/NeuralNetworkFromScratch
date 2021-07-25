# Input "layer"
# noinspection PyPep8Naming
import numpy as np


class Layer_Input:

    # Forward pass
    def forward(self, inputs: np.ndarray, training: bool):
        self.output = inputs
