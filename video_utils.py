import time
from pathlib import Path

import cv2
import numpy as np

from config import VIDEO_EXTS
from profiler import get_profile_log, timed, log_measure


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
    log = get_profile_log()
    video_path = Path(video_path)
    t0_total = time.perf_counter()

    if log:
        log.log("video.best_frame",
                 phase="start",
                 module="video_utils",
                 function="extract_best_frame_from_video",
                 filename=video_path.name,
                 file_path=str(video_path),
                 sample_count=sample_count)

    t_cap = time.perf_counter()
    cap = cv2.VideoCapture(str(video_path))
    log_measure(log, "video.cap_open",
                elapsed_ms=(time.perf_counter() - t_cap) * 1000.0,
                module="video_utils",
                filename=video_path.name)

    if not cap.isOpened():
        elapsed_ms = (time.perf_counter() - t0_total) * 1000.0
        if log:
            log.log("video.best_frame",
                     phase="end",
                     module="video_utils",
                     function="extract_best_frame_from_video",
                     elapsed_ms=round(elapsed_ms, 3),
                     filename=video_path.name,
                     status="read_failed")
        return None, "视频无法打开，建议视频文件名使用英文", "read_failed"

    t_fc = time.perf_counter()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log_measure(log, "video.frame_count",
                elapsed_ms=(time.perf_counter() - t_fc) * 1000.0,
                module="video_utils",
                filename=video_path.name,
                extra={"total_frames": total_frames})

    if total_frames <= 0:
        cap.release()
        elapsed_ms = (time.perf_counter() - t0_total) * 1000.0
        if log:
            log.log("video.best_frame",
                     phase="end",
                     module="video_utils",
                     function="extract_best_frame_from_video",
                     elapsed_ms=round(elapsed_ms, 3),
                     filename=video_path.name,
                     status="read_failed")
        return None, "无法读取视频帧数", "read_failed"

    start = int(total_frames * 0.15)
    end = int(total_frames * 0.85)

    if end <= start:
        start = 0
        end = total_frames - 1

    frame_indices = np.linspace(start, end, sample_count).astype(int)
    if log:
        log.log("video.frame_indices",
                 phase="measure",
                 module="video_utils",
                 filename=video_path.name,
                 extra={"indices": frame_indices.tolist()})

    best_frame = None
    best_area = 0.0
    best_index = None
    last_valid_frame = None

    for frame_index in frame_indices:
        t_seek = time.perf_counter()
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        seek_ms = (time.perf_counter() - t_seek) * 1000.0

        t_read = time.perf_counter()
        success, frame = cap.read()
        read_ms = (time.perf_counter() - t_read) * 1000.0

        if not success or frame is None:
            if log:
                log.log("video.sample_frame",
                         phase="measure",
                         module="video_utils",
                         filename=video_path.name,
                         frame_index=int(frame_index),
                         seek_ms=round(seek_ms, 3),
                         read_ms=round(read_ms, 3),
                         status="read_failed")
            continue

        last_valid_frame = frame.copy()

        t_detect = time.perf_counter()
        bboxes, _ = face_app.det_model.detect(frame)
        detect_ms = (time.perf_counter() - t_detect) * 1000.0

        if bboxes.shape[0] == 0:
            if log:
                log.log("video.sample_detect",
                         phase="measure",
                         module="video_utils",
                         filename=video_path.name,
                         frame_index=int(frame_index),
                         seek_ms=round(seek_ms, 3),
                         read_ms=round(read_ms, 3),
                         detect_ms=round(detect_ms, 3),
                         faces_count=0,
                         face_area=0.0,
                         detection_only=True)
            continue

        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        max_idx = int(np.argmax(areas))
        area = float(areas[max_idx])

        if log:
            log.log("video.sample_detect",
                     phase="measure",
                     module="video_utils",
                     filename=video_path.name,
                     frame_index=int(frame_index),
                     seek_ms=round(seek_ms, 3),
                     read_ms=round(read_ms, 3),
                     detect_ms=round(detect_ms, 3),
                     faces_count=int(bboxes.shape[0]),
                     face_area=round(area, 1),
                     detection_only=True)

        if area > best_area:
            best_area = area
            best_frame = frame.copy()
            best_index = int(frame_index)

    cap.release()

    if best_frame is None:
        status = "no_face" if last_valid_frame is not None else "read_failed"
        info = "视频中没有检测到可用人脸" if last_valid_frame is not None else "视频无法解码任何帧"
        elapsed_ms = (time.perf_counter() - t0_total) * 1000.0
        if log:
            log.log("video.best_frame",
                     phase="end",
                     module="video_utils",
                     function="extract_best_frame_from_video",
                     elapsed_ms=round(elapsed_ms, 3),
                     filename=video_path.name,
                     status=status)
        return (last_valid_frame, info, status) if last_valid_frame is not None else (None, info, status)

    elapsed_ms = (time.perf_counter() - t0_total) * 1000.0
    if log:
        log.log("video.best_frame",
                 phase="end",
                 module="video_utils",
                 function="extract_best_frame_from_video",
                 elapsed_ms=round(elapsed_ms, 3),
                 filename=video_path.name,
                 status="ok",
                 extra={"best_frame_index": best_index, "best_area": round(best_area, 1)})

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
    log = get_profile_log()
    video_path = Path(video_path)
    logs = []
    t0_total = time.perf_counter()

    if log:
        log.log("video.face_frames",
                 phase="start",
                 module="video_utils",
                 function="extract_face_frames_from_video",
                 filename=video_path.name,
                 file_path=str(video_path),
                 sample_count=sample_count,
                 max_frames=max_frames)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        elapsed_ms = (time.perf_counter() - t0_total) * 1000.0
        if log:
            log.log("video.face_frames",
                     phase="end",
                     module="video_utils",
                     function="extract_face_frames_from_video",
                     elapsed_ms=round(elapsed_ms, 3),
                     filename=video_path.name,
                     status="read_failed")
        return [], [f"视频无法打开: {video_path.name}，建议文件名使用英文"]

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        elapsed_ms = (time.perf_counter() - t0_total) * 1000.0
        if log:
            log.log("video.face_frames",
                     phase="end",
                     module="video_utils",
                     function="extract_face_frames_from_video",
                     elapsed_ms=round(elapsed_ms, 3),
                     filename=video_path.name,
                     status="read_failed")
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

        t_face = time.perf_counter()
        bboxes, _ = face_app.det_model.detect(frame)
        detect_ms = (time.perf_counter() - t_face) * 1000.0

        if bboxes.shape[0] == 0:
            if log:
                log.log("video.sample_detect",
                         phase="measure",
                         module="video_utils",
                         filename=video_path.name,
                         frame_index=int(frame_index),
                         detect_ms=round(detect_ms, 3),
                         faces_count=0,
                         detection_only=True)
            continue

        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        max_idx = int(np.argmax(areas))
        area = float(areas[max_idx])

        if log:
            log.log("video.sample_detect",
                     phase="measure",
                     module="video_utils",
                     filename=video_path.name,
                     frame_index=int(frame_index),
                     detect_ms=round(detect_ms, 3),
                     faces_count=int(bboxes.shape[0]),
                     detection_only=True)

        candidates.append(
            {
                "frame": frame.copy(),
                "frame_index": int(frame_index),
                "face_area": float(area),
            }
        )

    cap.release()

    if not candidates:
        elapsed_ms = (time.perf_counter() - t0_total) * 1000.0
        if log:
            log.log("video.face_frames",
                     phase="end",
                     module="video_utils",
                     function="extract_face_frames_from_video",
                     elapsed_ms=round(elapsed_ms, 3),
                     filename=video_path.name,
                     status="no_face",
                     extra={"candidates_count": 0})
        return [], [f"视频中没有检测到可用人脸: {video_path.name}"]

    # 按人脸面积从大到小排序，优先保留脸更清晰/更大的帧
    candidates.sort(key=lambda x: x["face_area"], reverse=True)

    selected = candidates[:max_frames]

    logs.append(
        f"{video_path.name}: 共找到 {len(candidates)} 个候选帧，保存 {len(selected)} 帧"
    )

    elapsed_ms = (time.perf_counter() - t0_total) * 1000.0
    if log:
        log.log("video.face_frames",
                 phase="end",
                 module="video_utils",
                 function="extract_face_frames_from_video",
                 elapsed_ms=round(elapsed_ms, 3),
                 filename=video_path.name,
                 status="ok",
                 extra={"candidates_count": len(candidates), "selected_count": len(selected)})

    return selected, logs
