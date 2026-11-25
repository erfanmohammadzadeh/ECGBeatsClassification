"""Train a 1D CNN for multilabel Challenge 2021 ECG classification."""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import callbacks, layers, models

parser = argparse.ArgumentParser(description="Train CNN on prepared Challenge 2021 records")
parser.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "dataset"),
                    help="prepared dataset directory")
parser.add_argument("--model-dir", default=os.path.join(os.path.dirname(__file__), "model"),
                    help="output model directory")
parser.add_argument("--epochs", type=int, default=1000, help="maximum epochs")
parser.add_argument("--batch-size", type=int, default=512,
                    help="records per training batch (default: 16; lower this if GPU memory is tight)")
args = parser.parse_args()
if args.batch_size < 1:
    parser.error("--batch-size must be at least 1")
try:
    sys.stdout.reconfigure(errors="replace")
except (AttributeError, ValueError):
    pass

def load(name):
    path = os.path.join(args.data, name + ".npy")
    if not os.path.isfile(path):
        raise SystemExit(f"Missing {path}; run prepare_data.py first")
    # ECG arrays are large; map them from disk instead of copying them into RAM.
    return np.load(path, mmap_mode=None if name == "classes" else "r",
                   allow_pickle=name == "classes")

X_train, X_test = load("X_train"), load("X_test")
y_train, y_test = load("y_train"), load("y_test")
classes = load("classes").astype(str)
if X_train.ndim != 3 or X_test.ndim != 3:
    raise SystemExit("Expected ECG inputs shaped (records, samples, leads); regenerate with prepare_data.py")
if y_train.ndim != 2 or y_train.shape[1] != len(classes):
    raise SystemExit("Expected multilabel targets shaped (records, classes); regenerate with prepare_data.py")
print(f"Train: {X_train.shape}, test: {X_test.shape}, classes: {len(classes)}")

# Let TensorFlow grow its GPU allocation as needed instead of reserving all VRAM.
for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

sequence_length, lead_count = X_train.shape[1:]
rng = np.random.default_rng(42)
indices = rng.permutation(len(X_train))
validation_count = max(1, int(len(indices) * 0.2))
validation_indices = indices[:validation_count]
training_indices = indices[validation_count:]
batch_size = args.batch_size

def batch_generator(array, targets, row_indices, shuffle):
    """Read only one small slice of the memory-mapped ECG files at a time."""
    order_rng = np.random.default_rng(42)
    while True:
        order = order_rng.permutation(row_indices) if shuffle else row_indices
        for start in range(0, len(order), batch_size):
            batch_indices = order[start:start + batch_size]
            yield np.asarray(array[batch_indices], dtype=np.float32), np.asarray(
                targets[batch_indices], dtype=np.float32)

def make_dataset(array, targets, row_indices, shuffle):
    dataset = tf.data.Dataset.from_generator(
        lambda: batch_generator(array, targets, row_indices, shuffle),
        output_signature=(
            tf.TensorSpec((None, sequence_length, lead_count), tf.float32),
            tf.TensorSpec((None, y_train.shape[1]), tf.float32),
        ),
    )
    return dataset.prefetch(1)

train_dataset = make_dataset(X_train, y_train, training_indices, shuffle=True)
validation_dataset = make_dataset(X_train, y_train, validation_indices, shuffle=False)
test_indices = np.arange(len(X_test))
test_dataset = make_dataset(X_test, y_test, test_indices, shuffle=False)
steps_per_epoch = int(np.ceil(len(training_indices) / batch_size))
validation_steps = int(np.ceil(len(validation_indices) / batch_size))
test_steps = int(np.ceil(len(test_indices) / batch_size))

model = models.Sequential([
    layers.Input(shape=X_train.shape[1:]),
    layers.Conv1D(32, 9, activation="relu", padding="same"),
    layers.BatchNormalization(), layers.MaxPooling1D(2), layers.Dropout(0.15),
    layers.Conv1D(64, 7, activation="relu", padding="same"),
    layers.BatchNormalization(), layers.MaxPooling1D(2), layers.Dropout(0.15),
    layers.Conv1D(128, 5, activation="relu", padding="same"),
    layers.BatchNormalization(), layers.GlobalAveragePooling1D(),
    layers.Dense(128, activation="relu"), layers.Dropout(0.3),
    layers.Dense(len(classes), activation="sigmoid"),
], name="Challenge2021_ECG_CNN")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
             loss="binary_crossentropy", metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy")])
model.summary()
os.makedirs(args.model_dir, exist_ok=True)
cb = [
    callbacks.EarlyStopping(monitor="val_loss", patience=100, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    callbacks.ModelCheckpoint(os.path.join(args.model_dir, "ecg_cnn_best.keras"),
                              monitor="val_loss", save_best_only=True, verbose=0),
    callbacks.CSVLogger(os.path.join(args.model_dir, "training_log.csv")),
]
history = model.fit(train_dataset, epochs=args.epochs, steps_per_epoch=steps_per_epoch,
                    validation_data=validation_dataset, validation_steps=validation_steps,
                    callbacks=cb, verbose=1)
test_loss, test_acc = model.evaluate(test_dataset, steps=test_steps, verbose=0)
y_pred = (model.predict(test_dataset, steps=test_steps, verbose=0) >= 0.5).astype(np.int32)
print(f"Test binary accuracy: {test_acc:.2%}; loss: {test_loss:.4f}")
print(classification_report(y_test, y_pred, target_names=classes, zero_division=0))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for metric, axis, title in [("loss", axes[0], "Loss"),
                            ("binary_accuracy", axes[1], "Binary accuracy")]:
    axis.plot(history.history[metric], label="train")
    axis.plot(history.history["val_" + metric], label="validation")
    axis.set_title(title); axis.set_xlabel("Epoch"); axis.legend(); axis.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(args.model_dir, "training_curves.png"), dpi=150)

cm = confusion_matrix(y_test.ravel(), y_pred.ravel())
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(cm, cmap="Blues")
ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No", "Yes"],
       yticklabels=["No", "Yes"], xlabel="Predicted", ylabel="Actual",
       title="Multilabel confusion matrix (all codes)")
for i in range(2):
    for j in range(2): ax.text(j, i, str(cm[i, j]), ha="center", va="center")
fig.tight_layout()
fig.savefig(os.path.join(args.model_dir, "confusion_matrix.png"), dpi=150)
model.save(os.path.join(args.model_dir, "ecg_cnn.keras"))
with open(os.path.join(args.model_dir, "results.json"), "w", encoding="utf-8") as f:
    json.dump({"test_binary_accuracy": float(test_acc), "test_loss": float(test_loss),
               "epochs_trained": len(history.history["loss"]), "classes": classes.tolist(),
               "input_shape": list(X_train.shape[1:])}, f, indent=2)
print(f"Saved model and reports under {args.model_dir}")
