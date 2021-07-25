import numpy as np
import nnfs
import os
import cv2
from typing import Tuple, List

from accuracies.categorical import Accuracy_Categorical
from activation_functions.relu import Activation_ReLU
from activation_functions.softmax import Activation_Softmax
from layers.layer_dense import Layer_Dense
from losses.loss_categorical_crossentropy import Loss_CategoricalCrossentropy
from model import Model
from optimizers.adam import Optimizer_Adam

nnfs.init()


# Loads a MNIST dataset
def load_mnist_dataset(dataset: str, path: str) -> Tuple[np.ndarray, np.ndarray]:
    # Scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))

    # Create lists for samples and labels
    X: List[np.ndarray] = []
    y: List[str] = []

    # For each label folder
    for label in labels:
        # And for each image in given folder
        files = os.listdir(os.path.join(path, dataset, label))
        files = files[:int(len(files) * 0.1)]
        for file in files:
            img_path = os.path.join(path, dataset, label, file)
            # Read the image and append it and a label to the lists
            X.append(cv2.imread(img_path, cv2.IMREAD_UNCHANGED))
            y.append(label)

    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')


# MNIST dataset (train + test)
def create_data_mnist(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Load both sets separately
    X_train, y_train = load_mnist_dataset(dataset='train', path=path)
    X_test, y_test = load_mnist_dataset(dataset='test', path=path)

    # And return all the data
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # Create dataset
    X_train, y_train, X_test, y_test = create_data_mnist(path='fashion_mnist_images')

    # Shuffle the training dataset
    keys = np.arange(y_train.shape[0])
    np.random.shuffle(keys)
    X_train = X_train[keys]
    y_train = y_train[keys]

    # Scale and reshape samples
    X_train = (X_train.reshape(X_train.shape[0], -1).astype(np.float32) - 127.5) / 127.5
    X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

    # Instantiate the model
    model = Model()

    # Add layers
    model.add(Layer_Dense(X_train.shape[1], 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 10))
    model.add(Activation_Softmax())

    # Set loss, optimizer and accuracies objects
    model.set(
        loss=Loss_CategoricalCrossentropy(),
        optimizer=Optimizer_Adam(decay=1e-3),
        accuracy=Accuracy_Categorical()
    )

    # Finalize the model
    model.finalize()

    # Train the model
    model.train(X_train, y_train, validation_data=(X_test, y_test),
                epochs=10, batch_size=128, print_every=100)
