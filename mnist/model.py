"""Model"""
import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import (
    InputLayer,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
    Dense
)


class SimpleCnnModel(keras.Model):
    """Simple Image Classification Network"""
    def __init__(self, n_classes: int = 10,
                 input_shape: list = [28, 28, 1]) -> None:
        """Simple Image Classification Network

        Args:
            n_classes (int, optional): Number of Classes. Defaults to 10.
            input_size (tuple, optional): Image Size(H, W, C). Defaults to MNist.
        """
        super().__init__()
        self.conv1 = Sequential(
            [   
                InputLayer(input_shape=input_shape),
                Conv2D(32, (3, 3), (1, 1), activation="relu"),
                MaxPooling2D()
            ],
            name="convolution-1"
        )
        self.conv2 = Sequential(
            [
                Conv2D(16, (3, 3), (1, 1), activation="relu"),
                MaxPooling2D()
            ],
            name="convolution-2"
        )
        self.conv3 = Sequential(
            [
                Conv2D(8, (3, 3), (1, 1), activation="relu"),
                MaxPooling2D(),
                Flatten()
            ],
            name="convolution-3"
        )
        self.fc = Sequential(
            [
                Dense(32, activation="relu"),
                BatchNormalization(),
                Dense(16, activation="relu"),
                BatchNormalization()
            ],
            name="fully-connected"
        )
        self.out = Dense(n_classes, activation="softmax", name="out")
    
    # TODO Inspect training, mask argument
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Training Step

        Args:
            inputs (tf.Tensor): Input Tensors
        """
        logits = self.conv1(inputs)
        logits = self.conv2(logits)
        logits = self.conv3(logits)
        logits = self.fc(logits)
        return self.out(logits)


class SimpleDenseModel(keras.Model):
    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.model = Sequential()
        self.model.add(Flatten())
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Dense(64, activation="tanh"))
        self.model.add(Dense(32, activation="sigmoid"))
        self.model.add(Dense(16, activation="relu"))
        self.model.add(Dense(n_classes, activation="softmax"))
        
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.model(inputs)
