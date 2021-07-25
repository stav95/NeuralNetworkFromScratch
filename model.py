from typing import List

import numpy as np

from accuracies.accuracy import Accuracy
from activation_and_loss.activation_softmax_loss_categorical_crossentropy import \
    Activation_Softmax_Loss_CategoricalCrossentropy
from layers.layer import Layer_Input
from activation_functions.softmax import Activation_Softmax

# Model class
# noinspection PyPep8Naming
from layers.layer_dense import Layer_Dense
from losses.loss import Loss
from losses.loss_categorical_crossentropy import Loss_CategoricalCrossentropy
from optimizers.optimizer import Optimizer


class Model:
    def __init__(self):
        # Create a list of network objects
        self.layers = []
        # Softmax classifier's output object
        self.softmax_classifier_output = None

    # Add objects to the model
    def add(self, layer):
        """
        :param layer: Layer_Dense or Activation Function
        :return: None
        """
        self.layers.append(layer)

    # Set loss, optimizer and accuracies
    def set(self, *, loss: Loss, optimizer: Optimizer, accuracy: Accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    # Finalize the model
    def finalize(self):
        # Create and set the input layer
        self.input_layer = Layer_Input()

        # Count all the objects
        layer_count = len(self.layers)

        # Iterate the objects
        for i in range(layer_count):
            # If it's the first layer,
            # the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]

            # All layers except for the first and the last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]

            # The last layer - the next object is the loss
            # Also let's save aside the reference to the last object
            # whose output is the model's output
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
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
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()

    # Train the model
    def train(self, X: np.ndarray,
              y: np.ndarray,
              *,
              epochs: int = 1,
              batch_size=None,
              print_every: int = 1,
              validation_data=None):

        # Initialize accuracies object
        self.accuracy.init(y)

        # Default value if batch size is not being set
        train_steps = 1

        X_val, y_val = None, None

        # If there is validation data passed,
        # set default number of steps for validation as well
        if validation_data is not None:
            validation_steps = 1

            # For better readability
            X_val, y_val = validation_data

        # Calculate number of steps
        if batch_size is not None:
            train_steps = X.shape[0] // batch_size
            # Dividing rounds down. If there are some remaining
            # data but not a full batch, this won't include it
            # Add `1` to include this not full batch
            if train_steps * batch_size < X.shape[0]:
                train_steps += 1

            if validation_data is not None:
                assert X_val is not None
                validation_steps = X_val.shape[0] // batch_size

                # Dividing rounds down. If there are some remaining
                # data but nor full batch, this won't include it
                # Add `1` to include this not full batch
                if validation_steps * batch_size < X_val.shape[0]:
                    validation_steps += 1

        # Main training loop
        for epoch in range(1, epochs + 1):

            # Print epoch number
            print(f'epoch: {epoch}')

            # Reset accumulated values in loss and accuracies objects
            self.loss.new_pass()
            self.accuracy.new_pass()

            # Iterate over steps
            for step in range(train_steps):

                # If batch size is not set -
                # train using one step and full dataset
                if batch_size is None:
                    batch_X = X
                    batch_y = y

                # Otherwise slice a batch
                else:
                    batch_X = X[step * batch_size:(step + 1) * batch_size]
                    batch_y = y[step * batch_size:(step + 1) * batch_size]

                batch_X: np.ndarray = batch_X
                batch_y: np.ndarray = batch_y

                # Perform the forward pass
                output = self.forward(X=batch_X, training=True)

                # Calculate loss
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss

                # Get predictions and calculate an accuracies
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                # Perform backward pass
                self.backward(output=output, y=batch_y)

                # Optimize (update parameters)
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
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
            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(f'training, ' +
                  f'acc: {epoch_accuracy:.3f}, ' +
                  f'loss: {epoch_loss:.3f} (' +
                  f'data_loss: {epoch_data_loss:.3f}, ' +
                  f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                  f'lr: {self.optimizer.current_learning_rate}')

            # If there is the validation data
            if validation_data is not None:

                # Reset accumulated values in loss
                # and accuracies objects
                self.loss.new_pass()
                self.accuracy.new_pass()

                # Iterate over steps
                for step in range(validation_steps):

                    # If batch size is not set -
                    # train using one step and full dataset
                    if batch_size is None:
                        batch_X = X_val
                        batch_y = y_val

                    # Otherwise slice a batch
                    else:
                        batch_X = X_val[step * batch_size:(step + 1) * batch_size]
                        batch_y = y_val[step * batch_size:(step + 1) * batch_size]

                    # Perform the forward pass
                    output = self.forward(batch_X, training=False)

                    # Calculate the loss
                    self.loss.calculate(output, batch_y)

                    # Get predictions and calculate an accuracies
                    predictions = self.output_layer_activation.predictions(output)
                    self.accuracy.calculate(predictions, batch_y)

                # Get and print validation loss and accuracies
                validation_loss = sum(self.loss.calculate_accumulated())
                validation_accuracy = self.accuracy.calculate_accumulated()

                # Print a summary
                print(f'validation, ' +
                      f'acc: {validation_accuracy:.3f}, ' +
                      f'loss: {validation_loss:.3f}')

    # Performs forward pass
    def forward(self, X: np.ndarray, training: bool) -> np.ndarray:

        # Call forward method on the input layer
        # this will set the output property that
        # the first layer in "prev" object is expecting
        self.input_layer.forward(inputs=X, training=training)

        # Call forward method of every object in a chain
        # Pass output of the previous object as a parameter
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        # "layer" is now the last object from the list,
        # return its output
        return self.layers[-1].output

    # Performs backward pass
    def backward(self, output: np.ndarray, y: np.ndarray):

        # If softmax classifier
        if self.softmax_classifier_output is not None:
            # First call backward method
            # on the combined activation/loss
            # this will set dinputs property
            self.softmax_classifier_output.backward(output, y)

            # Since we'll not call backward method of the last layer
            # which is Softmax activation
            # as we used combined activation/loss
            # object, let's set dinputs in this object
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            # Call backward method going through
            # all the objects but last
            # in reversed order passing dinputs as a parameter
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return

        # First call backward method on the loss
        # this will set dinputs property that the last
        # layer will try to access shortly
        self.loss.backward(output, y)

        # Call backward method going through all the objects
        # in reversed order passing dinputs as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
