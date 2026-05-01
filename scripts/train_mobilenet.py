import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = BASE_DIR / ".cache"
MPL_DIR = CACHE_DIR / "matplotlib"
XDG_DIR = CACHE_DIR / "xdg"
MPL_DIR.mkdir(parents=True, exist_ok=True)
XDG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_DIR))

IMG_SIZE = (224, 224)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
EPOCHS_STAGE1 = int(os.getenv("MB_EPOCHS_STAGE1", "5"))
EPOCHS_STAGE2 = int(os.getenv("MB_EPOCHS_STAGE2", "3"))

DATASET_DIR = Path(os.getenv("DATASET_DIR", str(BASE_DIR / "images" / "dataset")))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "mobilenet_best.keras")))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", str(BASE_DIR / "reports")))

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

train_dir = DATASET_DIR / "train"
val_dir = DATASET_DIR / "val"
test_dir = DATASET_DIR / "test"

for required_dir in [train_dir, val_dir, test_dir]:
    if not required_dir.exists():
        raise FileNotFoundError(f"Dossier dataset introuvable: {required_dir}")

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

class_names = train_ds.class_names
num_classes = len(class_names)
if num_classes < 2:
    raise RuntimeError("Il faut au moins deux classes/personnes pour entraîner le modèle.")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
])

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
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
    metrics=["accuracy"],
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(filepath=str(MODEL_PATH), monitor="val_loss", save_best_only=True),
]

print("=== Entraînement MobileNetV2 stage 1 ===")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    callbacks=callbacks,
)

print("=== Fine-tuning MobileNetV2 ===")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    callbacks=callbacks,
)

# Save final weights even if the last epoch was not the best validation checkpoint.
model.save(MODEL_PATH)

print("=== Évaluation ===")
y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
y_probs = model.predict(test_ds)
y_pred = np.argmax(y_probs, axis=1)

result = {
    "model": "mobilenet",
    "classes": class_names,
    "accuracy": float(np.mean(y_true == y_pred)),
    "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
    "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    "classification_report": classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    ),
}
summary = {
    "classes": class_names,
    "mobilenet": result,
}

with (REPORTS_DIR / "summary.json").open("w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2, ensure_ascii=False)
with (REPORTS_DIR / "mobilenet_metrics.json").open("w", encoding="utf-8") as file:
    json.dump(result, file, indent=2, ensure_ascii=False)

print(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"Modèle sauvegardé dans: {MODEL_PATH}")
