import numpy as np
import cv2
import os
from typing import Tuple, List
from accuracies.categorical import Accuracy_Categorical
from activation_functions.relu import Activation_ReLU
from activation_functions.softmax import Activation_Softmax
from layers.layer_dense import Layer_Dense
from losses.loss_categorical_crossentropy import Loss_CategoricalCrossentropy
from model import Model
from optimizers.adam import Optimizer_Adam


# noinspection PyPep8Naming
def load_fashion_mnist_dataset(dataset: str, path: str) -> Tuple[np.ndarray, np.ndarray]:
    X: List[np.ndarray] = []
    y: List[str] = []

    labels = os.listdir(os.path.join(path, dataset))

    for label in labels:
        dir_path = os.path.join(path, dataset, label)
        files_name = os.listdir(os.path.join(path, dataset, label))
        files = [os.path.join(dir_path, file) for file in files_name]
        files = files[: int(len(files) * 0.1)]

        X += list(map(lambda f: cv2.imread(f, cv2.IMREAD_UNCHANGED), files))
        y += [label] * len(files)

    return np.array(X), np.array(y).astype('uint8')


# noinspection PyPep8Naming,PyShadowingNames
def create_dataset_fashion_mnist(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, y_train = load_fashion_mnist_dataset(dataset='train', path=path)
    X_test, y_test = load_fashion_mnist_dataset(dataset='test', path=path)

    return X_train, y_train, X_test, y_test


def create_model() -> Model:
    m = Model()

    m.add(layer=Layer_Dense(n_inputs=X_train.shape[1], n_neurons=128))
    m.add(layer=Activation_ReLU())
    m.add(layer=Layer_Dense(n_inputs=128, n_neurons=256))
    m.add(layer=Activation_ReLU())
    m.add(layer=Layer_Dense(n_inputs=256, n_neurons=10))
    m.add(layer=Activation_Softmax())

    m.complie(optimizer=Optimizer_Adam(decay=1e-3),
              loss=Loss_CategoricalCrossentropy(),
              accuracy=Accuracy_Categorical())

    return m


# noinspection PyShadowingNames
def save_model_parameters(model: Model, filename: str):
    model_parameters = model.get_parameters()
    np.save(filename, model_parameters, allow_pickle=True)


# noinspection PyShadowingNames
def load_model_parameters(model: Model, filename: str):
    params = np.load(filename, allow_pickle=True)
    parameters = [tuple(params[i]) for i in range(params.shape[0])]
    model.set_parameters(parameters=parameters)


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = create_dataset_fashion_mnist(path='fashion_mnist_images')

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

    model.train(X=X_train,
                y=y_train,
                epochs=10,
                batch_size=128,
                validation_data=(X_test, y_test),
                print_every=100)

    # save_model_parameters(model=model, filename='ready_model_10.npy')

    # params = model_parameters = model.get_parameters()
    # [print(arr[0].shape, arr[1].shape) for arr in params]
    # model.evaluate(X_val=X_test, y_val=y_test, batch_size=128, print_evaluation=True)
    # model.plot_model_results()
    # model.plot_model_results(plot_training=True)
