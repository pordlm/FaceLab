from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path):
    path = Path(path)

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None

        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def write_image(path: Path, img) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower()
    if ext == "":
        ext = ".jpg"

    try:
        success, encoded = cv2.imencode(ext, img)
        if not success:
            return False

        encoded.tofile(str(path))
        return True
    except Exception:
        return False
