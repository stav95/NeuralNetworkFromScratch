import abc

import numpy as np

from layers.layer import Layer


class ActivationFunction(Layer, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

        self.dinputs: np.ndarray = np.array([])

    @abc.abstractmethod
    def predictions(self, outputs: np.ndarray):
        pass
