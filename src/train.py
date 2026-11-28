"""Train the 360-sample, two-lead, five-class Holter beat classifier."""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import callbacks, layers, models

HERE = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description="Train the Holter AAMI beat classifier")
parser.add_argument("--data", default=os.path.join(HERE, "dataset"), help="prepared beat dataset")
parser.add_argument("--model-dir", default=os.path.join(HERE, "model"), help="model output directory")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch-size", type=int, default=128)
args = parser.parse_args()

def load(name):
    path = os.path.join(args.data, name + ".npy")
    if not os.path.isfile(path):
        raise SystemExit(f"Missing {path}. Run prepare_data.py first.")
    return np.load(path, allow_pickle=name.startswith("records_"))

X_train, X_test = load("X_train"), load("X_test")
y_train, y_test = load("y_train"), load("y_test")
X_val, y_val = load("X_val"), load("y_val")
classes = load("classes").astype(str)
if X_train.ndim != 3 or X_train.shape[1:] != (360, 2):
    raise SystemExit(f"Holter model expects (records, 360, 2); got {X_train.shape}. Rebuild data.")
if not np.array_equal(classes, np.array(["F", "N", "S", "V"])):
    raise SystemExit(f"Unexpected AAMI class order: {classes.tolist()}")
print(f"Train beats: {len(X_train):,}; validation beats: {len(X_val):,}; "
      f"test beats: {len(X_test):,}; classes: {classes.tolist()}")
if len(X_train) == 0 or len(X_val) == 0:
    raise SystemExit("Training/validation split is empty. Rebuild the prepared dataset.")
print("Balanced training counts:", {
    str(classes[i]): int((y_train == i).sum()) for i in range(len(classes))
})

model = models.Sequential([
    layers.Input(shape=(360, 2)),
    layers.Conv1D(32, 5, padding="same", activation="relu"),
    layers.BatchNormalization(), layers.MaxPooling1D(2), layers.Dropout(0.15),
    layers.Conv1D(64, 5, padding="same", activation="relu"),
    layers.BatchNormalization(), layers.MaxPooling1D(2), layers.Dropout(0.15),
    layers.Conv1D(128, 3, padding="same", activation="relu"),
    layers.BatchNormalization(), layers.GlobalAveragePooling1D(),
    layers.Dense(64, activation="relu"), layers.Dropout(0.25),
    layers.Dense(len(classes), activation="softmax"),
], name="Holter_ECG_Beat_Classifier")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
os.makedirs(args.model_dir, exist_ok=True)
cb = [
    callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    callbacks.ModelCheckpoint(os.path.join(args.model_dir, "ecg_cnn_best.h5"),
                              monitor="val_loss", save_best_only=True, verbose=0),
    callbacks.CSVLogger(os.path.join(args.model_dir, "training_log.csv")),
]
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.epochs,
                    batch_size=args.batch_size, callbacks=cb,
                    shuffle=True, verbose=1)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
print(f"Test beat accuracy: {test_acc:.2%}; loss: {test_loss:.4f}")
print(classification_report(y_test, y_pred, labels=np.arange(len(classes)),
                            target_names=classes, zero_division=0))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for metric, ax, title in (("loss", axes[0], "Loss"), ("accuracy", axes[1], "Accuracy")):
    ax.plot(history.history[metric], label="train")
    ax.plot(history.history["val_" + metric], label="validation")
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(args.model_dir, "training_curves.png"), dpi=150)
cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(classes)))
fig, ax = plt.subplots(figsize=(7, 6))
ax.imshow(cm, cmap="Blues")
ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
       xticklabels=classes, yticklabels=classes, xlabel="Predicted beat class",
       ylabel="True beat class", title="Holter AAMI beat confusion matrix")
for row in range(len(classes)):
    for col in range(len(classes)):
        ax.text(col, row, str(cm[row, col]), ha="center", va="center")
fig.tight_layout()
fig.savefig(os.path.join(args.model_dir, "confusion_matrix.png"), dpi=150)

# The Holter converter expects this filename and five softmax outputs.
model.save(os.path.join(args.model_dir, "ecg_cnn.h5"))
np.save(os.path.join(args.model_dir, "classes.npy"), classes)
with open(os.path.join(args.model_dir, "classes.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(classes) + "\n")
with open(os.path.join(args.model_dir, "results.json"), "w", encoding="utf-8") as f:
    json.dump({"test_accuracy": float(test_acc), "test_loss": float(test_loss),
               "epochs_trained": len(history.history["loss"]),
               "input_shape": [360, 2], "classes": classes.tolist()}, f, indent=2)
print(f"Saved Holter-compatible model and class order to {args.model_dir}")
