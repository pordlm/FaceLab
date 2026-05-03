import pickle
from pathlib import Path

from config import GALLERY_PATH, EMBEDDING_CACHE_PATH


def load_pickle_safe(path: Path):
    path = Path(path)

    if not path.exists():
        return None, f"文件不存在: {path.name}"

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data, ""
    except Exception as e:
        return None, str(e)


def get_gallery_summary(gallery_path: Path = GALLERY_PATH):
    """
    读取 gallery.pkl。

    gallery.pkl 结构大概是：
    {
        "zhangsan": embedding向量,
        "lisi": embedding向量
    }
    """
    gallery, error = load_pickle_safe(gallery_path)

    if error:
        return {
            "exists": gallery_path.exists(),
            "error": error,
            "person_count": 0,
            "rows": []
        }

    rows = []

    gallery = gallery or {}

    for name, embedding in gallery.items():
        try:
            dim = len(embedding)
            preview = [round(float(x), 4) for x in embedding[:8]]
        except Exception:
            dim = "unknown"
            preview = []

        rows.append(
            {
                "person_name": name,
                "feature_dim": dim,
                "embedding_preview": preview
            }
        )

    return {
        "exists": True,
        "error": "",
        "person_count": len(rows),
        "rows": rows
    }


def get_embedding_cache_summary(cache_path: Path = EMBEDDING_CACHE_PATH):
    """
    读取 embedding_cache.pkl。

    embedding_cache.pkl 结构大概是：
    {
        "version": 1,
        "items": {
            "G:/facelab/dataset/zhangsan/001.jpg": {
                "person_name": "zhangsan",
                "ok": True,
                "embedding": ...
            }
        }
    }
    """
    cache, error = load_pickle_safe(cache_path)

    if error:
        return {
            "exists": cache_path.exists(),
            "error": error,
            "total_items": 0,
            "rows": []
        }

    cache = cache or {}
    items = cache.get("items", {})

    person_stats = {}

    for file_path, item in items.items():
        person_name = item.get("person_name", "unknown")

        if person_name not in person_stats:
            person_stats[person_name] = {
                "person_name": person_name,
                "cached_ok_images": 0,
                "cached_failed_images": 0,
                "total_cached_images": 0,
            }

        person_stats[person_name]["total_cached_images"] += 1

        if item.get("ok"):
            person_stats[person_name]["cached_ok_images"] += 1
        else:
            person_stats[person_name]["cached_failed_images"] += 1

    rows = list(person_stats.values())
    rows.sort(key=lambda x: x["person_name"])

    return {
        "exists": True,
        "error": "",
        "total_items": len(items),
        "rows": rows
    }
