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

import tensorflow as tf

base = BASE_DIR / "images" / "raw"
print("Dossier raw existe :", base.exists())

if base.exists():
    for p in base.iterdir():
        if p.is_dir():
            count = len(list(p.glob("*")))
            print(p.name, "->", count, "fichiers")
else:
    print("Chemin attendu :", base)

print("TensorFlow OK :", tf.__version__)
