import pickle
from pathlib import Path

import numpy as np

from config import (
    DATASET_DIR,
    GALLERY_PATH,
    EMBEDDING_CACHE_PATH,
    IMAGE_EXTS,
)
from image_io import read_image
from face_engine import extract_embedding, l2_normalize


CACHE_VERSION = 1


def get_file_signature(image_path: Path):
    """
    用文件大小 + 修改时间判断图片有没有变化。
    """
    stat = image_path.stat()

    return {
        "size": stat.st_size,
        "mtime": stat.st_mtime,
    }


def load_embedding_cache(cache_path: Path = EMBEDDING_CACHE_PATH):
    cache_path = Path(cache_path)

    if not cache_path.exists():
        return {
            "version": CACHE_VERSION,
            "items": {}
        }

    try:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)

        if not isinstance(cache, dict):
            return {
                "version": CACHE_VERSION,
                "items": {}
            }

        if cache.get("version") != CACHE_VERSION:
            return {
                "version": CACHE_VERSION,
                "items": {}
            }

        if "items" not in cache:
            cache["items"] = {}

        return cache

    except Exception:
        return {
            "version": CACHE_VERSION,
            "items": {}
        }


def save_embedding_cache(cache, cache_path: Path = EMBEDDING_CACHE_PATH):
    cache_path = Path(cache_path)

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)


def is_cache_valid(cache_item, image_path: Path, person_name: str):
    """
    判断缓存是否还能复用。
    条件：
    1. 文件大小没变
    2. 修改时间没变
    3. 所属人物文件夹没变
    """
    if cache_item is None:
        return False

    signature = get_file_signature(image_path)

    return (
        cache_item.get("size") == signature["size"]
        and cache_item.get("mtime") == signature["mtime"]
        and cache_item.get("person_name") == person_name
    )


def build_gallery(
    face_app,
    dataset_dir: Path = DATASET_DIR,
    gallery_path: Path = GALLERY_PATH,
    cache_path: Path = EMBEDDING_CACHE_PATH,
    force_rebuild: bool = False,
):
    """
    增量式建立人脸库。

    第一次：
        提取所有 dataset/ 图片的 embedding
        保存 embedding_cache.pkl
        保存 gallery.pkl

    之后：
        未变化的图片直接复用缓存
        新增或修改的图片才重新提取 embedding
    """
    dataset_dir = Path(dataset_dir)
    gallery_path = Path(gallery_path)
    cache_path = Path(cache_path)

    gallery = {}
    logs = []

    if not dataset_dir.exists():
        dataset_dir.mkdir(parents=True, exist_ok=True)
        logs.append("dataset 目录不存在，已自动创建。")
        return gallery, logs

    cache = load_embedding_cache(cache_path)

    if force_rebuild:
        cache = {
            "version": CACHE_VERSION,
            "items": {}
        }
        logs.append("已启用强制重建：忽略旧缓存。")

    cache_items = cache["items"]
    active_keys = set()

    person_embeddings = {}

    for person_dir in sorted(dataset_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name
        person_embeddings.setdefault(person_name, [])

        for image_path in sorted(person_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue

            image_path = image_path.resolve()
            cache_key = str(image_path)
            active_keys.add(cache_key)

            cached_item = cache_items.get(cache_key)

            if not force_rebuild and is_cache_valid(cached_item, image_path, person_name):
                if cached_item.get("ok") and cached_item.get("embedding") is not None:
                    emb = cached_item["embedding"]
                    person_embeddings[person_name].append(emb)
                    logs.append(f"复用缓存: {person_name} / {image_path.name}")
                else:
                    logs.append(f"复用失败记录: {person_name} / {image_path.name}，原因: {cached_item.get('reason')}")
                continue

            img = read_image(image_path)

            signature = get_file_signature(image_path)

            if img is None:
                cache_items[cache_key] = {
                    "person_name": person_name,
                    "size": signature["size"],
                    "mtime": signature["mtime"],
                    "ok": False,
                    "embedding": None,
                    "reason": "图片无法读取",
                }
                logs.append(f"无法读取图片: {image_path}")
                continue

            emb, face = extract_embedding(face_app, img)

            if emb is None:
                cache_items[cache_key] = {
                    "person_name": person_name,
                    "size": signature["size"],
                    "mtime": signature["mtime"],
                    "ok": False,
                    "embedding": None,
                    "reason": "没有检测到人脸",
                }
                logs.append(f"没有检测到人脸: {image_path}")
                continue

            cache_items[cache_key] = {
                "person_name": person_name,
                "size": signature["size"],
                "mtime": signature["mtime"],
                "ok": True,
                "embedding": emb,
                "reason": "",
            }

            person_embeddings[person_name].append(emb)
            logs.append(f"重新提取特征: {person_name} / {image_path.name}")

    # 删除缓存中已经不存在的图片
    old_keys = set(cache_items.keys())
    deleted_keys = old_keys - active_keys

    for key in deleted_keys:
        del cache_items[key]

    if deleted_keys:
        logs.append(f"已清理不存在图片的缓存: {len(deleted_keys)} 条")

    # 计算每个人的平均 embedding
    for person_name, embeddings in person_embeddings.items():
        if embeddings:
            mean_emb = np.mean(embeddings, axis=0)
            mean_emb = l2_normalize(mean_emb)

            gallery[person_name] = mean_emb
            logs.append(f"{person_name}: 已加入特征库，样本数 {len(embeddings)}")
        else:
            logs.append(f"{person_name}: 没有可用人脸样本")

    with open(gallery_path, "wb") as f:
        pickle.dump(gallery, f)

    save_embedding_cache(cache, cache_path)

    logs.append(f"图片级特征缓存已保存到: {cache_path}")
    logs.append(f"人员特征库已保存到: {gallery_path}")
    logs.append(f"共录入 {len(gallery)} 个人")

    return gallery, logs


def load_gallery(gallery_path: Path = GALLERY_PATH):
    gallery_path = Path(gallery_path)

    if not gallery_path.exists():
        raise FileNotFoundError("找不到 gallery.pkl，请先建立人脸库。")

    with open(gallery_path, "rb") as f:
        return pickle.load(f)
