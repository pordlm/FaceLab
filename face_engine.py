import numpy as np
from insightface.app import FaceAnalysis


def create_face_app(use_gpu: bool = False):
    if use_gpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = 0
    else:
        providers = ["CPUExecutionProvider"]
        ctx_id = -1

    app = FaceAnalysis(
        name="buffalo_l",
        providers=providers
    )
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
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
    faces = face_app.get(img)
    face = get_largest_face(faces)

    return face, faces


def extract_embedding_from_face(face):
    """
    从已经检测到的人脸对象中提取 embedding。
    """
    if face is None:
        return None

    emb = face.embedding.astype(np.float32)
    emb = l2_normalize(emb)

    return emb


def extract_embedding(face_app, img):
    """
    兼容旧代码的函数：
    检测最大人脸并提取 embedding。

    返回:
        embedding, face
    """
    face, faces = detect_largest_face(face_app, img)

    if face is None:
        return None, None

    emb = extract_embedding_from_face(face)

    return emb, face

