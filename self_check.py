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


def scan_output_files(output_dir: Path = OUTPUT_DIR) -> set[str]:
    """收集 output/ 子目录下所有文件名（用于与 input 对比）。"""
    if not output_dir.exists():
        return set()
    names = set()
    for label_dir in output_dir.iterdir():
        if not label_dir.is_dir():
            continue
        for f in label_dir.iterdir():
            if f.is_file():
                names.add(f.name)
    return names


def _get_expected_output_name(input_file: Path) -> str:
    """根据 input 文件名推导 output 中的预期文件名。视频输出原始视频文件。"""
    return input_file.name


def find_problems(
    input_files: list[Path],
    output_names: set[str],
    results: list[dict] | None,
) -> tuple[list[Path], list[Path]]:
    """
    找出问题文件。

    返回:
        (read_failed_files, missing_files)

    read_failed_files: 分类结果中 decision == "read_failed" 的文件（Path 指向 input/）
    missing_files: input 中有但 output 中找不到对应结果的文件（Path 指向 input/）
    """
    read_failed_files: list[Path] = []
    missing_files: list[Path] = []

    # 构建 filename → result 的映射（不含路径，只有文件名）
    result_map: dict[str, dict] = {}
    if results:
        for r in results:
            result_map[r["filename"]] = r

    for input_file in input_files:
        expected_name = _get_expected_output_name(input_file)
        result = result_map.get(input_file.name)

        # 检查是否为 read_failed
        if result and result.get("decision") == "read_failed":
            read_failed_files.append(input_file)

        # 检查 output 中是否存在对应文件
        if expected_name not in output_names:
            # 避免重复加入（read_failed 的文件可能同时也不在 output 中）
            if input_file not in read_failed_files:
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
        img, source_info = load_image_or_video_frame(file_path, face_app)

        if img is None:
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
    output_names = scan_output_files(output_dir)
    results = load_classification_results(output_dir)

    read_failed_files, missing_files = find_problems(
        input_files, output_names, results
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
        "output_count": len(output_names),
        "read_failed_count": len(read_failed_files),
        "missing_count": len(missing_files),
        "reclassify_success": success_list,
        "reclassify_failed": failed_list,
        "results_loaded": results is not None,
    }
