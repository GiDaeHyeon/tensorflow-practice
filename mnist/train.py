"""Train Module"""
from datetime import datetime

from keras.callbacks import EarlyStopping, TensorBoard

from model import SimpleCnnModel, SimpleDenseModel
from dataloader import get_mnist


# Model & Data
INPUT_SHAPE = [28, 28, 1]
MODEL = SimpleCnnModel(input_shape=INPUT_SHAPE)
# INPUT_SHAPE = [784,]
# MODEL = SimpleDenseModel()

(X_train, y_train), (X_test, y_test) = get_mnist()

# Callback
now = datetime.now().strftime("%Y%m%d")
EARLY_STOPPING = EarlyStopping(
    monitor="accuracy", min_delta=1e-2, patience=20, mode="max"
)
TENSORBOARD = TensorBoard(
    log_dir=f"logs/cnn/{now}",
)


if __name__ == "__main__":
    MODEL.compile(loss="categorical_crossentropy",
                optimizer="sgd", metrics=["accuracy"])
    MODEL.build([None] + INPUT_SHAPE)
    MODEL.summary()
    MODEL.fit(X_train, y_train, epochs=100, batch_size=128,
            validation_data=(X_test, y_test),
            callbacks=[EARLY_STOPPING, TENSORBOARD])
