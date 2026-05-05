from pathlib import Path

import cv2
import numpy as np

from config import VIDEO_EXTS
from face_engine import get_largest_face, get_face_area


def is_video_file(path: Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def extract_best_frame_from_video(video_path: Path, face_app, sample_count: int = 8):
    """
    从视频中抽取若干帧，选择检测到最大人脸的一帧。

    返回:
        (best_frame, info, status)
        status 取值:
            "ok"         — 成功检测到人脸
            "no_face"    — 视频可读取，但所有采样帧均未检测到人脸
            "read_failed" — 视频无法打开或无法读取帧数

    如果 status == "ok":
        best_frame 是最大人脸帧, info 是人脸信息
    如果 status == "no_face":
        best_frame 是最后成功读取的帧（用于输出到 no_face）, info 是说明
    如果 status == "read_failed":
        best_frame 是 None, info 是错误原因
    """
    video_path = Path(video_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None, "视频无法打开，建议视频文件名使用英文", "read_failed"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return None, "无法读取视频帧数", "read_failed"

    start = int(total_frames * 0.15)
    end = int(total_frames * 0.85)

    if end <= start:
        start = 0
        end = total_frames - 1

    frame_indices = np.linspace(start, end, sample_count).astype(int)

    best_frame = None
    best_area = 0.0
    best_index = None
    last_valid_frame = None

    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = cap.read()

        if not success or frame is None:
            continue

        last_valid_frame = frame.copy()

        faces = face_app.get(frame)
        face = get_largest_face(faces)

        if face is None:
            continue

        area = get_face_area(face)

        if area > best_area:
            best_area = area
            best_frame = frame.copy()
            best_index = int(frame_index)

    cap.release()

    if best_frame is None:
        if last_valid_frame is not None:
            return last_valid_frame, "视频中没有检测到可用人脸", "no_face"
        return None, "视频无法解码任何帧", "read_failed"

    return best_frame, f"已选择第 {best_index} 帧，人脸面积 {best_area:.0f}", "ok"


def extract_face_frames_from_video(
    video_path: Path,
    face_app,
    max_frames: int = 8,
    sample_count: int = 30,
):
    """
    从视频中抽取多张包含人脸的帧，用于建立人脸库。

    返回:
        frames, logs

    frames:
        [
            {
                "frame": frame,
                "frame_index": 123,
                "face_area": 45678.0
            }
        ]
    """
    video_path = Path(video_path)
    logs = []

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return [], [f"视频无法打开: {video_path.name}，建议文件名使用英文"]

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return [], [f"无法读取视频帧数: {video_path.name}"]

    start = int(total_frames * 0.10)
    end = int(total_frames * 0.90)

    if end <= start:
        start = 0
        end = total_frames - 1

    frame_indices = np.linspace(start, end, sample_count).astype(int)

    candidates = []

    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = cap.read()

        if not success or frame is None:
            continue

        faces = face_app.get(frame)
        face = get_largest_face(faces)

        if face is None:
            continue

        area = get_face_area(face)

        candidates.append(
            {
                "frame": frame.copy(),
                "frame_index": int(frame_index),
                "face_area": float(area),
            }
        )

    cap.release()

    if not candidates:
        return [], [f"视频中没有检测到可用人脸: {video_path.name}"]

    # 按人脸面积从大到小排序，优先保留脸更清晰/更大的帧
    candidates.sort(key=lambda x: x["face_area"], reverse=True)

    selected = candidates[:max_frames]

    logs.append(
        f"{video_path.name}: 共找到 {len(candidates)} 个候选帧，保存 {len(selected)} 帧"
    )

    return selected, logs
