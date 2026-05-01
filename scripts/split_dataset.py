from pathlib import Path
import random
import shutil

random.seed(42)

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "images" / "processed"
DATASET_DIR = BASE_DIR / "images" / "dataset"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def copy_files(files, target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, target_dir / f.name)

def main():
    if not PROCESSED_DIR.exists():
        raise FileNotFoundError(f"Dossier introuvable: {PROCESSED_DIR}")

    for split in ["train", "val", "test"]:
        (DATASET_DIR / split).mkdir(parents=True, exist_ok=True)

    for person_dir in PROCESSED_DIR.iterdir():
        if not person_dir.is_dir():
            continue

        images = [p for p in person_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
        random.shuffle(images)

        n = len(images)
        train_end = int(n * TRAIN_RATIO)
        val_end = train_end + int(n * VAL_RATIO)

        train_files = images[:train_end]
        val_files = images[train_end:val_end]
        test_files = images[val_end:]

        copy_files(train_files, DATASET_DIR / "train" / person_dir.name)
        copy_files(val_files, DATASET_DIR / "val" / person_dir.name)
        copy_files(test_files, DATASET_DIR / "test" / person_dir.name)

        print(f"{person_dir.name}: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

if __name__ == "__main__":
    main()
