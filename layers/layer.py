import numpy as np
import abc


class Layer(metaclass=abc.ABCMeta):
    def __init__(self):
        self.inputs: np.ndarray = np.array([])
        self.output: np.ndarray = np.array([])

        self.prev, self.next = None, None

    @abc.abstractmethod
    def forward(self, inputs: np.ndarray):
        pass

    @abc.abstractmethod
    def backward(self, dvalues: np.ndarray):
        pass
