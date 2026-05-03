import shutil
from pathlib import Path

from config import (
    INPUT_DIR,
    OUTPUT_DIR,
    GALLERY_PATH,
    IMAGE_EXTS,
    VIDEO_EXTS,
    DEFAULT_THRESHOLD,
)
from image_io import read_image, write_image
from face_engine import extract_embedding, cosine_similarity
from gallery_builder import load_gallery
from video_utils import is_video_file, extract_best_frame_from_video


def is_supported_file(path: Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS.union(VIDEO_EXTS)


def load_image_or_video_frame(file_path: Path, face_app):
    """
    图片：直接读取
    视频：自动抽取最佳人脸帧
    """
    file_path = Path(file_path)

    if is_video_file(file_path):
        frame, info = extract_best_frame_from_video(file_path, face_app)
        return frame, info

    img = read_image(file_path)

    if img is None:
        return None, "图片无法读取"

    return img, "图片读取成功"


def predict_person(embedding, gallery, threshold: float):
    best_name = "unknown"
    best_score = -1.0

    for name, known_emb in gallery.items():
        score = cosine_similarity(embedding, known_emb)

        if score > best_score:
            best_score = score
            best_name = name

    if best_score < threshold:
        return "unknown", best_score

    return best_name, best_score


def classify_all(
    face_app,
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
    gallery_path: Path = GALLERY_PATH,
    threshold: float = DEFAULT_THRESHOLD,
):
    """
    对 input/ 里的图片或视频进行分类。

    返回 results:
        [
          {
            "filename": "...",
            "label": "...",
            "score": 0.51,
            "source_info": "...",
            "output_path": "..."
          }
        ]
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    gallery = load_gallery(gallery_path)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for file_path in sorted(input_dir.iterdir()):
        if not file_path.is_file():
            continue

        if not is_supported_file(file_path):
            continue

        img, source_info = load_image_or_video_frame(file_path, face_app)

        if img is None:
            results.append(
                {
                    "filename": file_path.name,
                    "label": "read_failed",
                    "score": 0.0,
                    "decision": "read_failed",
                    "source_info": source_info,
                    "output_path": "",
                }
            )

            continue

        emb, face = extract_embedding(face_app, img)

        if face is None:
            label = "no_face"
            score = 0.0
            decision = "no_face"
        elif emb is None:
            label = "no_face"
            score = 0.0
            decision = "no_face"
        else:
            label, score = predict_person(emb, gallery, threshold)

            if label == "unknown":
                decision = "face_detected_unknown"
            else:
                decision = "face_detected_known"


        person_output_dir = output_dir / label
        person_output_dir.mkdir(parents=True, exist_ok=True)

        if is_video_file(file_path):
            dst_path = person_output_dir / f"{file_path.stem}_frame.jpg"
            write_image(dst_path, img)
        else:
            dst_path = person_output_dir / file_path.name
            shutil.copy2(file_path, dst_path)

        results.append(
            {
                "filename": file_path.name,
                "label": label,
                "score": float(score),
                "decision": decision,
                "source_info": source_info,
                "output_path": str(dst_path),
            }
        )


    return results
