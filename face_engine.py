import time

import numpy as np
from insightface.app import FaceAnalysis

from profiler import get_profile_log, timed, log_measure, is_profiling_enabled


def create_face_app(use_gpu: bool = False):
    log = get_profile_log()

    if use_gpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = 0
    else:
        providers = ["CPUExecutionProvider"]
        ctx_id = -1

    t0 = time.perf_counter()
    if log:
        log.log("create_face_app",
                 phase="start",
                 module="face_engine",
                 function="create_face_app",
                 use_gpu=use_gpu,
                 providers_requested=providers,
                 ctx_id_requested=ctx_id,
                 det_size=[640, 640])

    with timed(log, "face_analysis.init",
               module="face_engine",
               function="create_face_app",
               taskname="init"):
        app = FaceAnalysis(
            name="buffalo_l",
            providers=providers,
            allowed_modules=["detection", "recognition"]
        )

    with timed(log, "face_analysis.prepare",
               module="face_engine",
               function="create_face_app",
               taskname="prepare"):
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if log:
        log.log("create_face_app",
                 phase="end",
                 module="face_engine",
                 function="create_face_app",
                 elapsed_ms=round(elapsed_ms, 3),
                 use_gpu=use_gpu,
                 providers_requested=providers,
                 ctx_id_requested=ctx_id,
                 det_size=[640, 640])

    return app


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def get_largest_face(faces):
    if not faces:
        return None

    def area(face):
        x1, y1, x2, y2 = face.bbox
        return float((x2 - x1) * (y2 - y1))

    return max(faces, key=area)


def get_face_area(face) -> float:
    x1, y1, x2, y2 = face.bbox
    return float((x2 - x1) * (y2 - y1))


def detect_largest_face(face_app, img):
    """
    只做人脸检测，不提取身份。

    返回:
        face, faces
    """
    log = get_profile_log()
    t0 = time.perf_counter()
    faces = face_app.get(img)
    face = get_largest_face(faces)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if log:
        log.log("detect_largest_face",
                 phase="measure",
                 module="face_engine",
                 function="detect_largest_face",
                 elapsed_ms=round(elapsed_ms, 3),
                 faces_count=len(faces),
                 image_shape=list(img.shape) if img is not None else None)
    return face, faces


def extract_embedding_from_face(face):
    """
    从已经检测到的人脸对象中提取 embedding。
    """
    log = get_profile_log()
    if face is None:
        return None

    t0 = time.perf_counter()
    emb = face.embedding.astype(np.float32)
    emb = l2_normalize(emb)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if log:
        log.log("extract_embedding_from_face",
                 phase="measure",
                 module="face_engine",
                 function="extract_embedding_from_face",
                 elapsed_ms=round(elapsed_ms, 3))
    return emb


def extract_embedding(face_app, img):
    """
    兼容旧代码的函数：
    检测最大人脸并提取 embedding。

    返回:
        embedding, face
    """
    log = get_profile_log()
    t0 = time.perf_counter()
    face, faces = detect_largest_face(face_app, img)

    if face is None:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if log:
            log.log("extract_embedding",
                     phase="measure",
                     module="face_engine",
                     function="extract_embedding",
                     elapsed_ms=round(elapsed_ms, 3),
                     status="no_face",
                     image_shape=list(img.shape) if img is not None else None)
        return None, None

    emb = extract_embedding_from_face(face)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if log:
        log.log("extract_embedding",
                 phase="measure",
                 module="face_engine",
                 function="extract_embedding",
                 elapsed_ms=round(elapsed_ms, 3),
                 status="ok",
                 image_shape=list(img.shape) if img is not None else None)
    return emb, face

