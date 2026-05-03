from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
GALLERY_PATH = BASE_DIR / "gallery.pkl"
EMBEDDING_CACHE_PATH = BASE_DIR / "embedding_cache.pkl"
ZIP_PATH = BASE_DIR / "classified_output.zip"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

DEFAULT_THRESHOLD = 0.45


def ensure_dirs():
    DATASET_DIR.mkdir(exist_ok=True)
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
