import abc

from layers.layer_dense import Layer_Dense


class Optimizer(metaclass=abc.ABCMeta):
    def __init__(self, learning_rate: float = 1., decay: float = 0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    @abc.abstractmethod
    def pre_update_params(self):
        pass

    @abc.abstractmethod
    def update_params(self, layer: Layer_Dense):
        pass

    @abc.abstractmethod
    def post_update_params(self):
        pass
