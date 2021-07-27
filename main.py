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
# noinspection PyPep8Naming
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
        # files = files[:int(len(files) * 0.1)]
        for file in files:
            img_path = os.path.join(path, dataset, label, file)
            # Read the image and append it and a label to the lists
            X.append(cv2.imread(img_path, cv2.IMREAD_UNCHANGED))
            y.append(label)

    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')


# MNIST dataset (train + test)
# noinspection PyPep8Naming,PyShadowingNames
def create_data_mnist(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Load both sets separately
    X_train, y_train = load_mnist_dataset(dataset='train', path=path)
    X_test, y_test = load_mnist_dataset(dataset='test', path=path)

    # And return all the data
    return X_train, y_train, X_test, y_test


def create_model() -> Model:
    # Instantiate the model
    m = Model()

    # Add layers
    m.add(layer=Layer_Dense(n_inputs=X_train.shape[1], n_neurons=128))
    m.add(layer=Activation_ReLU())
    m.add(layer=Layer_Dense(n_inputs=128, n_neurons=128))
    m.add(layer=Activation_ReLU())
    m.add(layer=Layer_Dense(n_inputs=128, n_neurons=10))
    m.add(layer=Activation_Softmax())

    # Set loss, optimizer and accuracies objects
    m.complie(optimizer=Optimizer_Adam(decay=1e-3),
              loss=Loss_CategoricalCrossentropy(),
              accuracy=Accuracy_Categorical())

    return m


def save_model_parameters(model: Model, filename: str):
    model_parameters = model.get_parameters()
    np.save(filename, model_parameters, allow_pickle=True)


def load_model_parameters(model: Model, filename: str):
    params = np.load(filename, allow_pickle=True)
    parameters = [tuple(params[i]) for i in range(params.shape[0])]
    model.set_parameters(parameters=parameters)


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
    #
    model = create_model()

    # load_model_parameters(model=model, filename='ready_model_100.npy')

    # Train the model
    model.train(X=X_train,
                y=y_train,
                epochs=10,
                batch_size=128,
                validation_data=(X_test, y_test),
                print_every=100)

    save_model_parameters(model=model, filename='ready_model_10.npy')

    # model.evaluate(X_val=X_test, y_val=y_test, batch_size=128, print_evaluation=True)
    model.plot_model_results()
    model.plot_model_results(plot_training=True)
