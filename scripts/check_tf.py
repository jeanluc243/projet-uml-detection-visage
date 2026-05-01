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

print("TensorFlow version :", tf.__version__)
print("GPU disponibles :", tf.config.list_physical_devices("GPU"))
print("CPU disponibles :", tf.config.list_physical_devices("CPU"))
