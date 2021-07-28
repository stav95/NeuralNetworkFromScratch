import math
from typing import List, Tuple

import numpy as np

from accuracies.accuracy import Accuracy
from activation_and_loss.activation_softmax_loss_categorical_crossentropy import \
    Activation_Softmax_Loss_CategoricalCrossentropy
from layers.layer import Layer
from layers.layer_input import Layer_Input
from activation_functions.softmax import Activation_Softmax

# Model class
# noinspection PyPep8Naming
from layers.layer_dense import Layer_Dense
from losses.loss import Loss
from losses.loss_categorical_crossentropy import Loss_CategoricalCrossentropy
from optimizers.optimizer import Optimizer
import matplotlib.pyplot as plt


# noinspection PyUnboundLocalVariable,PyPep8Naming
class Model:
    def __init__(self):
        # Create a list of network objects
        self.layers = []
        # Softmax classifier's output object
        # noinspection PyTypeChecker
        self.softmax_classifier_output: Activation_Softmax_Loss_CategoricalCrossentropy = None
        self.any_layer_has_regularizer = False

    # Add objects to the model
    def add(self, layer: Layer):
        """
        :param layer: Layer_Dense or Activation Function
        :return: None
        """
        self.layers.append(layer)

    # Set loss, optimizer and accuracies
    def complie(self, optimizer: Optimizer, loss: Loss, accuracy: Accuracy):
        self.optimizer = optimizer
        self.loss = loss
        self.accuracy = accuracy

        # Create and set the input layer
        self.input_layer = Layer_Input()

        # Count all the objects
        layer_count = len(self.layers)

        layers: List[Layer] = self.layers
        # Iterate the objects
        for i in range(layer_count):
            # If it's the first layer,
            # the previous layer object is the input layer
            if i == 0:
                layers[i].prev = self.input_layer
                layers[i].next = self.layers[i + 1]

            # All layers except for the first and the last
            elif i < layer_count - 1:
                layers[i].prev = self.layers[i - 1]
                layers[i].next = self.layers[i + 1]

            # The last layer - the next object is the loss
            # Also let's save aside the reference to the last object
            # whose output is the model's output
            else:
                layers[i].prev = self.layers[i - 1]
                layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

        self.trainable_layers: List[Layer_Dense] = [_ for _ in self.layers if isinstance(_, Layer_Dense)]

        # Update loss object with trainable layers
        self.loss.remember_trainable_layers(self.trainable_layers)

        # If output activation is Softmax and
        # loss function is Categorical Cross-Entropy
        # create an object of combined activation
        # and loss function containing
        # faster gradient calculation
        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossentropy):
            # Create an object of combined activation
            # and loss functions
            # self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()
            pass

        for layer in self.trainable_layers:
            if layer.weight_regularizer_l1 > 0 or layer.weight_regularizer_l2 > 0 \
                    or layer.bias_regularizer_l1 > 0 or layer.bias_regularizer_l2 > 0:
                self.any_layer_has_regularizer = True
                break

    # Train the model
    def train(self,
              X: np.ndarray,
              y: np.ndarray,
              epochs: int = 1,
              batch_size: int = 0,
              validation_data: Tuple[np.ndarray, np.ndarray] = None,
              print_every: int = 1):

        train_steps = math.ceil(X.shape[0] / batch_size) if batch_size > 0 else 1

        self.training_acc, self.val_acc = [], []
        self.training_loss, self.val_loss = [], []

        for epoch in range(1, epochs + 1):
            print(f'epoch: {epoch}')

            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                if batch_size == 0:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step * batch_size:(step + 1) * batch_size]
                    batch_y = y[step * batch_size:(step + 1) * batch_size]

                batch_X: np.ndarray = batch_X
                batch_y: np.ndarray = batch_y

                # Perform the forward pass
                output = self.forward(X=batch_X)

                # Calculate loss
                data_loss, regularization_loss = self.loss.calculate(
                    output=output,
                    y=batch_y,
                    include_regularization=self.any_layer_has_regularizer)

                loss = data_loss + regularization_loss

                # Get predictions and calculate an accuracies
                predictions = self.output_layer_activation.predictions(outputs=output)
                accuracy = self.accuracy.calculate(predictions=predictions, y=batch_y)

                # Perform backward pass
                self.backward(output=output, y=batch_y)

                # Optimize (update parameters)
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer=layer)
                self.optimizer.post_update_params()

                # Print a summary
                if not step % print_every or step == train_steps - 1:
                    print(f'step: {step}, ' +
                          f'acc: {accuracy:.3f}, ' +
                          f'loss: {loss:.3f} (' +
                          f'data_loss: {data_loss:.3f}, ' +
                          f'reg_loss: {regularization_loss:.3f}), ' +
                          f'lr: {self.optimizer.current_learning_rate}')

            # Get and print epoch loss and accuracies
            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated_loss(
                include_regularization=self.any_layer_has_regularizer)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated_accuracy()

            print(f'training, ' +
                  f'acc: {epoch_accuracy:.3f}, ' +
                  f'loss: {epoch_loss:.3f} (' +
                  f'data_loss: {epoch_data_loss:.3f}, ' +
                  f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                  f'lr: {self.optimizer.current_learning_rate}')

            self.training_acc.append(epoch_accuracy)
            self.training_loss.append(epoch_loss)

            # If there is the validation data
            if validation_data is not None:
                validation_accuracy, validation_loss = self.evaluate(*validation_data, batch_size=batch_size)

                # Print a summary
                print(f'validation, ' +
                      f'acc: {validation_accuracy:.3f}, ' +
                      f'loss: {validation_loss:.3f}')

                self.val_acc.append(validation_accuracy)
                self.val_loss.append(validation_loss)

    # Performs forward pass
    def forward(self, X: np.ndarray) -> np.ndarray:

        # Call forward method on the input layer
        # this will set the output property that
        # the first layer in "prev" object is expecting
        self.input_layer.forward(inputs=X)

        layers: List[Layer] = self.layers
        # Call forward method of every object in a chain
        # Pass output of the previous object as a parameter
        for layer in layers:
            layer.forward(inputs=layer.prev.output)

        # "layer" is now the last object from the list,
        # return its output
        return self.layers[-1].output

    # Performs backward pass
    def backward(self, output: np.ndarray, y: np.ndarray):

        # If softmax classifier
        if self.softmax_classifier_output is not None:
            # First call backward method
            # on the combined activation/loss
            # this will set dinputs property)
            self.softmax_classifier_output.backward(dvalues=output, y_true=y)

            # Since we'll not call backward method of the last layer
            # which is Softmax activation
            # as we used combined activation/loss
            # object, let's set dinputs in this object
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            layers: List[Layer] = self.layers
            # Call backward method going through
            # all the objects but last
            # in reversed order passing dinputs as a parameter
            for layer in reversed(layers[:-1]):
                layer.backward(dvalues=layer.next.dinputs)

            return

        # First call backward method on the loss
        # this will set dinputs property that the last
        # layer will try to access shortly
        self.loss.backward(dvalues=output, y_true=y)

        layers: List[Layer] = self.layers
        # Call backward method going through all the objects
        # in reversed order passing dinputs as a parameter
        for layer in reversed(layers):
            layer.backward(dvalues=layer.next.dinputs)

    def evaluate(self,
                 X_val: np.ndarray,
                 y_val: np.ndarray,
                 batch_size: int = 0,
                 print_evaluation: bool = False) -> Tuple[float, float]:
        validation_steps = 1
        # Calculate number of steps
        if batch_size > 0:
            validation_steps = X_val.shape[0] // batch_size
        # Dividing rounds down. If there are some remaining
        # data, but not a full batch, this won't include it
        # Add `1` to include this not full minibatch
        if validation_steps * batch_size < X_val.shape[0]:
            validation_steps += 1
        # Reset accumulated values in loss
        # and accuracy objects
        self.loss.new_pass()
        self.accuracy.new_pass()
        # Iterate over steps
        for step in range(validation_steps):
            # train using one step and full dataset
            if batch_size == 0:
                batch_X = X_val
                batch_y = y_val
            # Otherwise slice a batch
            else:
                batch_X = X_val[step * batch_size:(step + 1) * batch_size]
                batch_y = y_val[step * batch_size:(step + 1) * batch_size]
            # Perform the forward pass
            output = self.forward(X=batch_X)
            # Calculate the loss
            self.loss.calculate(output=output, y=batch_y)
            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(outputs=output)
            self.accuracy.calculate(predictions=predictions, y=batch_y)

        validation_loss = sum(self.loss.calculate_accumulated_loss())
        validation_accuracy = self.accuracy.calculate_accumulated_accuracy()

        if not print_evaluation:
            return validation_accuracy, validation_loss
        # Print a summary
        print(f'validation, ' +
              f'acc: {validation_accuracy:.3f}, ' +
              f'loss: {validation_loss:.3f}')

    def get_parameters(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        # Create a list for parameters
        parameters = []
        # Iterable trainable layers and get their parameters
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
        # Return a list
        return parameters

    def set_parameters(self, parameters: List[Tuple[np.ndarray, np.ndarray]]):

        # Iterate over the parameters and layers
        # and update each layers with each set of the parameters
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)

    # Plot Model Results
    def plot_model_results(self, plot_training: bool = False):
        ndigits = 4

        if plot_training:
            if hasattr(self, 'training_acc') and hasattr(self, 'training_loss'):
                plt.plot(self.training_acc, label=f'Training Accuracy - {round(max(self.training_acc), ndigits)}')
                plt.plot(self.training_loss, label=f'Training Loss - {round(min(self.training_loss), ndigits)}')
                plt.legend()
                plt.show()
        else:
            if hasattr(self, 'val_acc') and hasattr(self, 'val_loss'):
                plt.plot(self.val_acc, label=f'Validation Accuracy - {round(max(self.val_acc), ndigits)}')
                plt.plot(self.val_loss, label=f'Validation Loss - {round(min(self.val_loss), ndigits)}')
                plt.legend()
                plt.show()
