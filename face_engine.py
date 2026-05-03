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


def extract_embedding(face_app, img):
    """
    返回:
        embedding, face
    如果没有检测到人脸:
        None, None
    """
    faces = face_app.get(img)
    face = get_largest_face(faces)

    if face is None:
        return None, None

    emb = face.embedding.astype(np.float32)
    emb = l2_normalize(emb)

    return emb, face
