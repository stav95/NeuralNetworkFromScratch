import numpy as np
from layers.layer_dense import Layer_Dense
from optimizers.optimizer import Optimizer


# noinspection PyPep8Naming
class Optimizer_Adam(Optimizer):
    def __init__(self,
                 learning_rate: float = 0.001,
                 decay: float = 0.,
                 epsilon: float = 1e-7,
                 beta_1: float = 0.9,
                 beta_2: float = 0.999):
        super().__init__(learning_rate=learning_rate, decay=decay)

        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer: Layer_Dense):
        layer.add_variables(add_cache=True, add_momentums=True)

        # Update momentum  with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        # Get corrected momentum, self.iteration is 0 at first pass and we need to start with 1 here
        weight_momentums = layer.weight_momentums / (1 - np.power(self.beta_1, self.iterations + 1))
        bias_momentums = layer.bias_momentums / (1 - np.power(self.beta_1, self.iterations + 1))

        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * np.power(layer.dweights, 2)
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * np.power(layer.dbiases, 2)

        # Get corrected cache
        weight_cache = layer.weight_cache / (1 - np.power(self.beta_2, self.iterations + 1))
        bias_cache = layer.bias_cache / (1 - np.power(self.beta_2, self.iterations + 1))

        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums / (np.sqrt(weight_cache) + self.epsilon)

        layer.biases += -self.current_learning_rate * bias_momentums / (np.sqrt(bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1
