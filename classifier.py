import json
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

    返回:
        (img, info, status)
        status 取值: "ok" / "no_face" / "read_failed"
    """
    file_path = Path(file_path)

    if is_video_file(file_path):
        frame, info, status = extract_best_frame_from_video(file_path, face_app)
        return frame, info, status

    img = read_image(file_path)

    if img is None:
        return None, "图片无法读取", "read_failed"

    return img, "图片读取成功", "ok"


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
            "decision": "...",
            "source_info": "...",
            "output_path": "...",
            "frame_path": "..."
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

        img, source_info, status = load_image_or_video_frame(file_path, face_app)

        if img is None or status == "no_face":
            if status == "no_face" and img is not None:
                # 视频可读取但未检测到人脸：输出抽帧图片到 no_face
                label = "no_face"
                decision = "no_face"
                person_output_dir = output_dir / label
                person_output_dir.mkdir(parents=True, exist_ok=True)
                frame_dst = person_output_dir / f"{file_path.stem}_frame.jpg"
                write_image(frame_dst, img)
                results.append(
                    {
                        "filename": file_path.name,
                        "label": label,
                        "score": 0.0,
                        "decision": decision,
                        "source_info": source_info,
                        "output_path": str(frame_dst),
                        "frame_path": "",
                    }
                )
                continue

            # 真读取失败（status == "read_failed" 或无图的 no_face）
            results.append(
                {
                    "filename": file_path.name,
                    "label": "read_failed",
                    "score": 0.0,
                    "decision": "read_failed",
                    "source_info": source_info,
                    "output_path": "",
                    "frame_path": "",
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
            # 保存抽帧图片
            frame_dst = person_output_dir / f"{file_path.stem}_frame.jpg"
            write_image(frame_dst, img)

            if face is not None and emb is not None:
                # 人脸检测成功：同时复制原始视频
                video_dst = person_output_dir / file_path.name
                shutil.copy2(file_path, video_dst)
                result_output_path = str(video_dst)
                result_frame_path = str(frame_dst)
            else:
                # no_face：只保留抽帧图片
                result_output_path = str(frame_dst)
                result_frame_path = ""
        else:
            dst_path = person_output_dir / file_path.name
            shutil.copy2(file_path, dst_path)
            result_output_path = str(dst_path)
            result_frame_path = ""

        results.append(
            {
                "filename": file_path.name,
                "label": label,
                "score": float(score),
                "decision": decision,
                "source_info": source_info,
                "output_path": result_output_path,
                "frame_path": result_frame_path,
            }
        )


    results_path = output_dir / "classification_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results
