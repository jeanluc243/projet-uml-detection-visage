import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "mobilenet_best.keras")))
RAW_DIR = BASE_DIR / "images" / "raw"
PROCESSED_DIR = BASE_DIR / "images" / "processed"
DATASET_DIR = BASE_DIR / "images" / "dataset"


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=BASE_DIR, check=True)


def has_images(path: Path) -> bool:
    return path.exists() and any(
        file.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        for file in path.rglob("*")
        if file.is_file()
    )


if MODEL_PATH.exists():
    print(f"Modèle déjà présent: {MODEL_PATH}", flush=True)
    raise SystemExit(0)

if not has_images(RAW_DIR):
    raise FileNotFoundError(
        f"Aucune image trouvée dans {RAW_DIR}. Ajoute images/raw/<personne>/*.jpg dans GitHub."
    )

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
print(f"Modèle absent. Entraînement en ligne vers: {MODEL_PATH}", flush=True)

# Rebuild generated datasets from the raw images committed in Git.
run([sys.executable, "scripts/extract_faces.py"])
if not has_images(PROCESSED_DIR):
    raise RuntimeError("Aucun visage extrait. Vérifie la qualité des images dans images/raw/.")

run([sys.executable, "scripts/split_dataset.py"])
if not has_images(DATASET_DIR / "train"):
    raise RuntimeError("Dataset d'entraînement vide après split_dataset.py.")

run([sys.executable, "scripts/train_mobilenet.py"])

if not MODEL_PATH.exists():
    raise RuntimeError(f"L'entraînement est terminé mais le modèle n'existe pas: {MODEL_PATH}")

print(f"Bootstrap terminé. Modèle disponible: {MODEL_PATH}", flush=True)
