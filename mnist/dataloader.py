"""DataLoader"""
import tensorflow as tf
from keras.datasets import mnist

def get_mnist():
    """Get Mnist Dataset"""
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    y_train = tf.one_hot(y_train, 10)
    y_test = tf.one_hot(y_test, 10)
    return (X_train, y_train), (X_test, y_test)
