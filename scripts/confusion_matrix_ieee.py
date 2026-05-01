import argparse
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = BASE_DIR / ".cache"
MPL_DIR = CACHE_DIR / "matplotlib"
XDG_DIR = CACHE_DIR / "xdg"

MPL_DIR.mkdir(parents=True, exist_ok=True)
XDG_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf


def has_supported_images(directory: Path) -> bool:
    if not directory.exists() or not directory.is_dir():
        return False

    allowed_suffixes = {".bmp", ".gif", ".jpeg", ".jpg", ".png"}
    return any(path.suffix.lower() in allowed_suffixes for path in directory.rglob("*"))


def resolve_repo_path(path_str: str, fallback: Path | None = None) -> Path:
    path = Path(path_str)
    candidates = []

    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(BASE_DIR / path)
        candidates.append(path)
        if fallback is not None:
            candidates.append(fallback)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def resolve_dataset_dir(path_str: str, fallback: Path) -> Path:
    requested = resolve_repo_path(path_str)
    if has_supported_images(requested):
        return requested
    if has_supported_images(fallback):
        return fallback
    return requested


def load_test_dataset(test_dir, img_size=(224, 224), batch_size=32):
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )
    class_names = test_ds.class_names
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    return test_ds, class_names


def get_predictions(model, test_ds):
    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    y_probs = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_probs, axis=1)
    return y_true, y_pred


def plot_confusion_matrix_ieee(
    cm,
    class_names,
    title="Confusion Matrix",
    save_path="confusion_matrix_ieee.png",
    normalize=False,
    dpi=300
):
    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm_display = np.nan_to_num(cm_display)
    else:
        cm_display = cm

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10
    })

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Count" if not normalize else "Ratio", rotation=90)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Ground Truth",
        xlabel="Predicted Label",
        title=title
    )

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    threshold = cm_display.max() / 2.0 if cm_display.size > 0 else 0.5
    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            value = cm_display[i, j]
            text = f"{value:.2f}" if normalize else f"{int(value)}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=10,
                fontweight="bold"
            )

    ax.spines[:].set_visible(False)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .keras model")
    parser.add_argument("--test_dir", required=True, help="Path to dataset/test")
    parser.add_argument("--output_dir", default="reports/confusion", help="Output folder")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--normalize", action="store_true", help="Normalize confusion matrix")
    args = parser.parse_args()

    model_path = resolve_repo_path(args.model)
    test_dir = resolve_dataset_dir(args.test_dir, BASE_DIR / "images" / "dataset" / "test")
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = model_path.stem

    print(f"\nLoading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print(f"Loading test dataset from: {test_dir}")
    test_ds, class_names = load_test_dataset(
        test_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size
    )

    print("Running predictions...")
    y_true, y_pred = get_predictions(model, test_ds)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm_path = output_dir / f"{model_name}_confusion_matrix_ieee.png"
    report_path = output_dir / f"{model_name}_classification_report.json"

    title = f"{model_name.upper()} - Confusion Matrix"
    plot_confusion_matrix_ieee(
        cm=cm,
        class_names=class_names,
        title=title,
        save_path=str(cm_path),
        normalize=args.normalize
    )

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nSaved confusion matrix to: {cm_path}")
    print(f"Saved classification report to: {report_path}")


if __name__ == "__main__":
    main()
