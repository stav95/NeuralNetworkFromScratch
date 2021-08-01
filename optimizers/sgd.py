import numpy as np
from layers.layer_dense import Layer_Dense
from optimizers.optimizer import Optimizer


# noinspection PyPep8Naming
class Optimizer_SGD(Optimizer):
    def __init__(self,
                 learning_rate: float = 1.,
                 decay: float = 0.,
                 momentum: float = 0.):
        super().__init__(learning_rate=learning_rate, decay=decay)

        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer: Layer_Dense):
        # If we use momentum
        if self.momentum:
            layer.add_variables(add_cache=False, add_momentums=True)

            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with current gradients
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        # Vanilla SGD updates (as before momentum update)
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1
