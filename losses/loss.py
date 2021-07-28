from typing import List, Tuple

import numpy as np
import abc

# Common loss class
# noinspection PyPep8Naming
from layers.layer_dense import Layer_Dense


class Loss(metaclass=abc.ABCMeta):
    def __init__(self):
        self.dinputs = np.empty([])
        self.accumulated_loss = 0
        self.accumulated_count = 0
        self.trainable_layers: List[Layer_Dense] = []

    @abc.abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def backward(self, dvalues: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        # Clip data to prevent division by 0
        return np.clip(dvalues, 1e-7, 1)

    # Regularization loss calculation
    def regularization_loss(self) -> float:
        # 0 by default
        regularization_loss = 0

        # Calculate regularization loss
        # iterate all trainable layers
        for layer in self.trainable_layers:

            # L1 regularization - weights
            # calculate only when factor greater than 0
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

            # L2 regularization - weights
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

            # L1 regularization - biases
            # calculate only when factor greater than 0
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

            # L2 regularization - biases
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss

    # Set/remember trainable layers
    def remember_trainable_layers(self, trainable_layers: List[Layer_Dense]):
        self.trainable_layers = trainable_layers

    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self,
                  output: np.ndarray,
                  y: np.ndarray,
                  include_regularization: bool = False) -> Tuple[float, float]:

        # Calculate sample losses
        batch_losses = self.forward(y_pred=output, y_true=y)

        # Calculate mean loss
        batch_loss_mean = batch_losses.mean()

        # Add accumulated sum of losses and sample count
        self.accumulated_loss += np.sum(batch_losses)
        self.accumulated_count += batch_losses.shape[0]

        # If just data loss - return it
        if not include_regularization:
            return batch_loss_mean, 0

        # Return the data and regularization losses
        return batch_loss_mean, self.regularization_loss()

    # Calculates accumulated loss
    def calculate_accumulated_loss(self, include_regularization=False) -> Tuple[float, float]:

        # Calculate mean loss
        data_loss = self.accumulated_loss / self.accumulated_count

        # If just data loss - return it
        if not include_regularization:
            return data_loss, 0

        # Return the data and regularization losses
        return data_loss, self.regularization_loss()

    # Reset variables for accumulated loss
    def new_pass(self):
        self.accumulated_loss = 0
        self.accumulated_count = 0
