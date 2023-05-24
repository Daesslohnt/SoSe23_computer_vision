import keras as k
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


X = np.load(os.path.join("..", "daten", "X.npy"))
y = np.load(os.path.join("..", "daten", "y.npy"))

X = np.reshape(X, (X.shape[0], 21, 2, 1))
y = np.reshape(y, (y.shape[0], 8))


def make_model(input_shape):
    input_layer = k.layers.Input(input_shape)

    conv = k.layers.Conv2D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv = k.layers.BatchNormalization()(conv)
    conv = k.layers.MaxPooling2D(pool_size=3, padding="same")(conv)

    conv = k.layers.Conv2D(filters=32, kernel_size=3, padding="same")(conv)
    conv = k.layers.BatchNormalization()(conv)
    conv = k.layers.ReLU()(conv)

    conv = k.layers.Conv2D(filters=16, kernel_size=3, padding="same")(conv)
    conv = k.layers.BatchNormalization()(conv)
    conv = k.layers.ReLU()(conv)

    conv = k.layers.Conv2D(filters=8, kernel_size=2, padding="same")(conv)
    conv = k.layers.BatchNormalization()(conv)
    conv = k.layers.ReLU()(conv)

    gap = k.layers.GlobalAvgPool2D()(conv)
    gap = k.layers.Dense(100)(gap)

    output_layer = k.layers.Dense(8, activation="softmax")(gap)

    return k.models.Model(inputs=input_layer, outputs=output_layer)

model = make_model(X.shape[1:])

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
model.compile(
    optimizer="sgd",
    loss=k.losses.CategoricalCrossentropy(),
    metrics=["categorical_accuracy", "Recall"]
)

callbacks = [
    k.callbacks.ModelCheckpoint(
        "best_model.h5", save_best_only=True, monitor="val_loss"
    ),
    k.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=50, verbose=1
    ),
    k.callbacks.EarlyStopping(
        monitor="val_loss", patience=50, verbose=1
    )
]

history = model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=5,
    validation_split=0.15,
    verbose=1,
    callbacks=callbacks
)

plt.figure()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig("loss.png")

plt.figure()
plt.plot(history.history["categorical_accuracy"])
plt.plot(history.history["val_categorical_accuracy"])
plt.title("categorical_accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.savefig("categorical_accuracy.png")
