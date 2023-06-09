import keras as k
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt


X = np.load(os.path.join("..", "daten", "X.npy"))
y = np.load(os.path.join("..", "daten", "y.npy"))

X = np.reshape(X, (X.shape[0], 21, 2, 1))
y = np.reshape(y, (y.shape[0], 8))


def make_model(input_shape):
    input_layer = k.layers.Input(input_shape)

    # TODO: try different architectures
    conv = k.layers.Conv2D(32, 3, activation='relu', padding="same")(input_layer)
    conv = k.layers.Conv2D(64, 3, activation='relu', padding="same")(conv)
    conv = k.layers.Flatten()(conv)
    fc = k.layers.Dense(128, activation="relu")(conv)
    fc = k.layers.Dense(64, activation="tanh")(fc)
    fc = k.layers.Dense(32, activation="sigmoid")(fc)
    fc = k.layers.Dense(16, activation="tanh")(fc)

    output_layer = k.layers.Dense(8, activation="softmax")(fc)

    return k.models.Model(inputs=input_layer, outputs=output_layer)

model = make_model(X.shape[1:])

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
model.compile(
    optimizer="adam",
    loss=k.losses.CategoricalCrossentropy(),
    metrics=["Recall", "categorical_accuracy"]
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
    epochs=30,
    validation_split=0.15,
    verbose=1,
    callbacks=callbacks
)

preds = model.predict(X_test)

for i in range(len(preds)):
    max_index = np.argmax(preds[i])
    for j in range(8):
        preds[i][j] = 1 if j == max_index else 0


p_left, p_double, p_right, p_up, p_down, p_wheel, p_hold, p_default = np.array(preds).T
t_left, t_double, t_right, t_up, t_down, t_wheel, t_hold, t_default = y_test.T
print("Sänsitivität:")
print("left:", recall_score(t_left, p_left))
print("double:", recall_score(t_double, p_double))
print("right:", recall_score(t_right, p_right))
print("up:", recall_score(t_up, p_up))
print("down:", recall_score(t_down, p_down))
print("wheel:", recall_score(t_wheel, p_wheel))
print("hold:", recall_score(t_hold, p_hold))
print("default:", recall_score(t_default, p_default))

plt.figure()
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="validation")
plt.title("Loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.savefig("loss.png")

plt.figure()
plt.plot(history.history["recall"], label="train")
plt.plot(history.history["val_recall"], label="validation")
plt.title("Recall")
plt.xlabel("epoch")
plt.ylabel("recall")
plt.legend()
plt.savefig("recall.png")

plt.figure()
plt.plot(history.history["categorical_accuracy"], label="train")
plt.plot(history.history["val_categorical_accuracy"], label="validation")
plt.title("categorical_accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.savefig("categorical_accuracy.png")

"""
    Sänsitivität:
    left: 1.0
    double: 1.0
    right: 1.0
    up: 1.0
    down: 1.0
    wheel: 0.0
    hold: 1.0
    default: 1.0
"""