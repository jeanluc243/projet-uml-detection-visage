import os
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

BASE_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = BASE_DIR / ".cache"
MPL_DIR = CACHE_DIR / "matplotlib"
XDG_DIR = CACHE_DIR / "xdg"

MPL_DIR.mkdir(parents=True, exist_ok=True)
XDG_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_DIR))

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CNN_EPOCHS = 20
MB_EPOCHS_STAGE1 = 12
MB_EPOCHS_STAGE2 = 8

DATASET_DIR = BASE_DIR / "images" / "dataset"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

train_dir = DATASET_DIR / "train"
val_dir = DATASET_DIR / "val"
test_dir = DATASET_DIR / "test"

# -----------------------------
# DATASET
# -----------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
])

# -----------------------------
# CALLBACKS
# -----------------------------
def build_callbacks(model_name: str):
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / f"{model_name}.keras"),
            monitor="val_loss",
            save_best_only=True
        )
    ]

# -----------------------------
# MODEL 1: CNN CLASSIQUE
# -----------------------------
def build_cnn():
    model = keras.Sequential([
        layers.Input(shape=(224, 224, 3)),
        data_augmentation,
        layers.Rescaling(1.0 / 255),

        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# -----------------------------
# MODEL 2: MOBILENETV2
# -----------------------------
def build_mobilenet():
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model, base_model

# -----------------------------
# PLOTS
# -----------------------------
def plot_history(history, title, prefix):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="train_accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.title(f"{title} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{prefix}_accuracy.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title(f"{title} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{prefix}_loss.png", dpi=180)
    plt.close()

# -----------------------------
# EVALUATION
# -----------------------------
def evaluate_model(model, model_name):
    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    y_probs = model.predict(test_ds)
    y_pred = np.argmax(y_probs, axis=1)

    acc = float(np.mean(y_true == y_pred))
    precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)

    result = {
        "model": model_name,
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": cm,
        "classification_report": report
    }

    with open(REPORTS_DIR / f"{model_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result

# -----------------------------
# TRAIN CNN
# -----------------------------
print("\n=== Entraînement CNN ===")
cnn_model = build_cnn()
cnn_history = cnn_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=CNN_EPOCHS,
    callbacks=build_callbacks("cnn_best")
)
plot_history(cnn_history, "CNN", "cnn")
cnn_result = evaluate_model(cnn_model, "cnn")

# -----------------------------
# TRAIN MOBILENET - STAGE 1
# -----------------------------
print("\n=== Entraînement MobileNetV2 - Stage 1 ===")
mobilenet_model, base_model = build_mobilenet()
mb_history_1 = mobilenet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=MB_EPOCHS_STAGE1,
    callbacks=build_callbacks("mobilenet_stage1_best")
)
plot_history(mb_history_1, "MobileNetV2 Stage 1", "mobilenet_stage1")

# -----------------------------
# FINETUNING
# -----------------------------
print("\n=== Fine-tuning MobileNetV2 ===")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

mobilenet_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

mb_history_2 = mobilenet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=MB_EPOCHS_STAGE2,
    callbacks=build_callbacks("mobilenet_best")
)
plot_history(mb_history_2, "MobileNetV2 Fine-tuning", "mobilenet_stage2")
mb_result = evaluate_model(mobilenet_model, "mobilenet")

# -----------------------------
# SUMMARY
# -----------------------------
summary = {
    "classes": class_names,
    "cnn": cnn_result,
    "mobilenet": mb_result
}

with open(REPORTS_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n=== Résumé final ===")
print(json.dumps(summary, indent=2, ensure_ascii=False))
