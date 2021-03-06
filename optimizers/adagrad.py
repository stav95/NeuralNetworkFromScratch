import numpy as np
from layers.layer_dense import Layer_Dense
from optimizers.optimizer import Optimizer


# noinspection PyPep8Naming
class Optimizer_Adagrad(Optimizer):
    def __init__(self,
                 learning_rate: float = 1.,
                 decay: float = 0.,
                 epsilon: float = 1e-7):
        super().__init__(learning_rate=learning_rate, decay=decay)

        self.epsilon = epsilon

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer: Layer_Dense):
        layer.add_variables(add_cache=True, add_momentums=False)

        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1
