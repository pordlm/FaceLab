import json
import shutil
from pathlib import Path

from config import (
    INPUT_DIR,
    OUTPUT_DIR,
    GALLERY_PATH,
    DEFAULT_THRESHOLD,
)
from image_io import write_image
from face_engine import extract_embedding
from gallery_builder import load_gallery
from classifier import (
    is_supported_file,
    load_image_or_video_frame,
    predict_person,
)
from video_utils import is_video_file


def load_classification_results(output_dir: Path = OUTPUT_DIR) -> list | None:
    """读取 output/classification_results.json，不存在则返回 None。"""
    results_path = output_dir / "classification_results.json"
    if not results_path.exists():
        return None
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def scan_input_files(input_dir: Path = INPUT_DIR) -> list[Path]:
    """收集 input/ 下所有支持的图片和视频文件。"""
    if not input_dir.exists():
        return []
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and is_supported_file(p)
    )


def count_json_output_files(results: list[dict] | None) -> int:
    """统计 classification_results.json 中 output_path 非空的条目数。"""
    if not results:
        return 0
    return sum(1 for r in results if r.get("output_path"))


def _resolve_path(raw_path: str) -> Path:
    """将 JSON 中的路径解析为绝对路径。兼容 Windows 绝对路径和相对路径。"""
    if not raw_path:
        return Path()
    p = Path(raw_path)
    if p.is_absolute():
        return p
    # 相对路径：相对于仓库根目录解析
    from config import BASE_DIR
    return (BASE_DIR / p).resolve()


def find_problems(
    input_files: list[Path],
    results: list[dict] | None,
) -> tuple[list[Path], list[Path]]:
    """
    以 classification_results.json 为权威来源找出问题文件。

    返回:
        (read_failed_files, missing_files)

    read_failed_files: decision == "read_failed" 的条目（直接作为重试候选，不验证路径）
    missing_files: JSON 中无记录，或 output_path/frame_path 指向的文件不存在
    """
    read_failed_files: list[Path] = []
    missing_files: list[Path] = []

    # 构建 filename → result 映射
    result_map: dict[str, dict] = {}
    if results:
        for r in results:
            result_map[r["filename"]] = r

    for input_file in input_files:
        entry = result_map.get(input_file.name)

        # 条件 1：JSON 中无此文件记录 → missing
        if entry is None:
            missing_files.append(input_file)
            continue

        decision = entry.get("decision", "")

        # 条件 2：read_failed → 直接作为重试候选，不验证路径
        if decision == "read_failed":
            read_failed_files.append(input_file)
            continue

        # 条件 3：非 read_failed — 验证 output_path / frame_path 存在性
        output_path = entry.get("output_path", "")
        frame_path = entry.get("frame_path", "")

        output_missing = bool(output_path) and not _resolve_path(output_path).exists()
        frame_missing = bool(frame_path) and not _resolve_path(frame_path).exists()

        if output_missing or frame_missing:
            missing_files.append(input_file)

    return read_failed_files, missing_files


def reclassify_files(
    file_paths: list[Path],
    face_app,
    gallery: dict,
    threshold: float,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[list[str], list[str], list[dict]]:
    """
    对指定的文件列表重新执行分类。

    返回:
        (success_list, failed_list, reclassify_results) — 文件名列表 + 完整结果字典列表
    """
    success_list: list[str] = []
    failed_list: list[str] = []
    reclassify_results: list[dict] = []

    for file_path in file_paths:
        img, source_info, status = load_image_or_video_frame(file_path, face_app)

        if img is None or status == "no_face":
            if status == "no_face" and img is not None:
                # 视频可读取但未检测到人脸：输出抽帧图片到 no_face
                label = "no_face"
                person_output_dir = output_dir / label
                person_output_dir.mkdir(parents=True, exist_ok=True)
                frame_dst = person_output_dir / f"{file_path.stem}_frame.jpg"
                frame_ok = write_image(frame_dst, img)
                if not frame_ok:
                    failed_list.append(file_path.name)
                    continue
                success_list.append(file_path.name)
                reclassify_results.append(
                    {
                        "filename": file_path.name,
                        "label": label,
                        "score": 0.0,
                        "decision": "no_face",
                        "source_info": source_info,
                        "output_path": str(frame_dst),
                        "frame_path": "",
                    }
                )
                continue

            # 真读取失败
            failed_list.append(file_path.name)
            continue

        emb, face = extract_embedding(face_app, img)

        if face is None or emb is None:
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
            frame_ok = write_image(frame_dst, img)

            if not frame_ok:
                failed_list.append(file_path.name)
                continue

            if label != "no_face":
                # 人脸检测成功：同时复制原始视频
                video_dst = person_output_dir / file_path.name
                try:
                    shutil.copy2(file_path, video_dst)
                except Exception:
                    failed_list.append(file_path.name)
                    continue
                result_output_path = str(video_dst)
                result_frame_path = str(frame_dst)
            else:
                # no_face：只保留抽帧图片
                result_output_path = str(frame_dst)
                result_frame_path = ""

            success_list.append(file_path.name)
            reclassify_results.append(
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
        else:
            dst_path = person_output_dir / file_path.name
            try:
                shutil.copy2(file_path, dst_path)
            except Exception:
                failed_list.append(file_path.name)
                continue

            success_list.append(file_path.name)
            reclassify_results.append(
                {
                    "filename": file_path.name,
                    "label": label,
                    "score": float(score),
                    "decision": decision,
                    "source_info": source_info,
                    "output_path": str(dst_path),
                    "frame_path": "",
                }
            )

    return success_list, failed_list, reclassify_results


def _merge_and_save_results(
    output_dir: Path,
    reclassify_results: list[dict],
) -> None:
    """
    将重分类结果合并到 output/classification_results.json。

    以 filename 为 key 覆盖旧条目，保留未受影响的条目不变。
    """
    results_path = output_dir / "classification_results.json"

    existing_results = load_classification_results(output_dir)
    if existing_results is None:
        existing_results = []

    # 构建 filename → result 的映射
    merged: dict[str, dict] = {}
    for r in existing_results:
        merged[r["filename"]] = r

    # 用重分类结果覆盖
    for r in reclassify_results:
        merged[r["filename"]] = r

    # 写回 JSON
    merged_list = list(merged.values())
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(merged_list, f, ensure_ascii=False, indent=2)


def run_self_check(
    face_app,
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
    gallery_path: Path = GALLERY_PATH,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """
    执行自检主流程。

    返回字典:
    {
        "input_count": int,           # input 文件总数
        "output_count": int,          # output 文件总数
        "read_failed_count": int,     # read_failed 数量
        "missing_count": int,         # 遗漏文件数量
        "reclassify_success": list,   # 重分类成功的文件名
        "reclassify_failed": list,    # 重分类失败的文件名
        "results_loaded": bool,       # 是否成功加载了分类结果 JSON
    }
    """
    input_files = scan_input_files(input_dir)
    results = load_classification_results(output_dir)

    read_failed_files, missing_files = find_problems(
        input_files, results
    )

    # 合并需要重试的文件（去重）
    to_retry = list(dict.fromkeys(read_failed_files + missing_files))

    gallery = load_gallery(gallery_path)

    success_list, failed_list, reclassify_results = reclassify_files(
        to_retry, face_app, gallery, threshold, output_dir
    )

    if reclassify_results:
        _merge_and_save_results(output_dir, reclassify_results)

    return {
        "input_count": len(input_files),
        "output_count": count_json_output_files(results),
        "read_failed_count": len(read_failed_files),
        "missing_count": len(missing_files),
        "reclassify_success": success_list,
        "reclassify_failed": failed_list,
        "results_loaded": results is not None,
    }
