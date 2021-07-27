import numpy as np

from layers.layer_dense import Layer_Dense
from optimizers.optimizer import Optimizer


# Adagrad optimizer
# noinspection PyPep8Naming
class Optimizer_Adagrad(Optimizer):

    # Initialize optimizer - set settings
    def __init__(self,
                 learning_rate: float = 1.,
                 decay: float = 0.,
                 epsilon: float = 1e-7):
        super().__init__(learning_rate=learning_rate, decay=decay)

        self.epsilon = epsilon

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer: Layer_Dense):

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1
