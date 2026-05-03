from pkl_viewer import get_gallery_summary, get_embedding_cache_summary
import shutil
import zipfile
from pathlib import Path
from image_io import write_image
from video_utils import is_video_file, extract_face_frames_from_video
import streamlit as st

from config import (
    DATASET_DIR,
    INPUT_DIR,
    OUTPUT_DIR,
    ZIP_PATH,
    IMAGE_EXTS,
    VIDEO_EXTS,
    DEFAULT_THRESHOLD,
    ensure_dirs,
)
from face_engine import create_face_app
from gallery_builder import build_gallery
from classifier import classify_all


@st.cache_resource
def load_cached_face_app(use_gpu: bool):
    return create_face_app(use_gpu=use_gpu)


def save_uploaded_files(uploaded_files, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for uploaded_file in uploaded_files:
        file_path = target_dir / uploaded_file.name

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        saved_paths.append(file_path)

    return saved_paths


def save_known_files_for_person(uploaded_files, target_dir: Path, face_app):
    """
    注册人员时使用：
    - 图片：直接保存到 dataset/人名/
    - 视频：抽取多张有人脸的帧，保存成 jpg 到 dataset/人名/
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    logs = []
    saved_count = 0

    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix.lower()

        temp_path = target_dir / f"_temp_{uploaded_file.name}"

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if is_video_file(temp_path):
            frames, frame_logs = extract_face_frames_from_video(
                temp_path,
                face_app,
                max_frames=8,
                sample_count=30
            )

            logs.extend(frame_logs)

            for idx, item in enumerate(frames):
                frame_index = item["frame_index"]
                frame = item["frame"]

                frame_name = f"{Path(uploaded_file.name).stem}_frame_{frame_index}_{idx}.jpg"
                frame_path = target_dir / frame_name

                ok = write_image(frame_path, frame)

                if ok:
                    saved_count += 1
                    logs.append(f"已保存视频抽帧: {frame_name}")
                else:
                    logs.append(f"保存视频抽帧失败: {frame_name}")

            temp_path.unlink(missing_ok=True)

        else:
            final_path = target_dir / uploaded_file.name

            # 如果刚才临时文件就是图片，移动成正式文件
            if final_path.exists():
                final_path.unlink()

            temp_path.rename(final_path)
            saved_count += 1
            logs.append(f"已保存图片: {uploaded_file.name}")

    return saved_count, logs


def clear_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)


def make_output_zip():
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in OUTPUT_DIR.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(OUTPUT_DIR))

    return ZIP_PATH


def list_people_dataset():
    people = []

    if not DATASET_DIR.exists():
        return people

    for person_dir in sorted(DATASET_DIR.iterdir()):
        if not person_dir.is_dir():
            continue

        count = len(
            [
                p for p in person_dir.iterdir()
                if p.suffix.lower() in IMAGE_EXTS
            ]
        )

        people.append((person_dir.name, count))

    return people


def main():
    ensure_dirs()

    st.set_page_config(
        page_title="Face Access Demo",
        layout="wide"
    )

    st.title("本地人脸门禁识别 Demo")
    st.caption("InsightFace + Streamlit，本地实验使用。")

    with st.sidebar:
        st.header("设置")

        use_gpu = st.checkbox("使用 GPU", value=False)

        threshold = st.slider(
            "识别阈值",
            min_value=0.20,
            max_value=0.80,
            value=DEFAULT_THRESHOLD,
            step=0.01
        )

        st.info(
            "阈值越高越严格，陌生人更容易进入 unknown；"
            "阈值越低越宽松，但更容易误认。"
        )

        if st.button("清空 input"):
            clear_dir(INPUT_DIR)
            st.success("已清空 input")

        if st.button("清空 output"):
            clear_dir(OUTPUT_DIR)
            st.success("已清空 output")

    face_app = load_cached_face_app(use_gpu)

    tab1, tab2, tab3, tab4 = st.tabs(
    ["1. 建立人脸库", "2. 识别图片/视频", "3. 查看结果", "4. 特征库状态"]
    )


    with tab1:
        st.subheader("上传已知人员照片")

        person_name = st.text_input(
            "人物名称",
            placeholder="例如 zhangsan / lisi / alice"
        )

        known_files = st.file_uploader(
            "上传这个人的照片或视频。视频会自动抽取多帧加入人脸库。",
            type=["jpg", "jpeg", "png", "webp", "mp4", "avi", "mov", "mkv"],
            accept_multiple_files=True
        )


        if st.button("保存到 dataset"):
            if not person_name.strip():
                st.error("请先输入人物名称。")
            elif not known_files:
                st.error("请先上传照片或视频。")
            else:
                target_dir = DATASET_DIR / person_name.strip()

                saved_count, logs = save_known_files_for_person(
                    known_files,
                    target_dir,
                    face_app
                )

                st.success(f"已保存 {saved_count} 张可用于建库的图片到 {target_dir}")

                with st.expander("查看保存日志"):
                    for log in logs:
                        st.write(log)


        st.divider()

        force_rebuild = st.checkbox("强制重新提取所有特征", value=False)

        if st.button("生成人脸库 gallery.pkl"):
            gallery, logs = build_gallery(
                face_app,
                force_rebuild=force_rebuild
            )

            st.success(f"人脸库生成完成，共 {len(gallery)} 个人。")

            with st.expander("查看处理日志"):
                for log in logs:
                    st.write(log)


            st.success(f"人脸库生成完成，共 {len(gallery)} 个人。")

            with st.expander("查看处理日志"):
                for log in logs:
                    st.write(log)

        st.divider()

        st.subheader("当前 dataset")

        people = list_people_dataset()

        if people:
            for name, count in people:
                st.write(f"- {name}: {count} 张")
        else:
            st.write("还没有已知人员照片。")

    with tab2:
        st.subheader("上传待识别图片或视频")

        input_files = st.file_uploader(
            "支持图片和视频。视频会自动抽取一帧进行识别。",
            type=["jpg", "jpeg", "png", "webp", "mp4", "avi", "mov", "mkv"],
            accept_multiple_files=True
        )

        if st.button("保存到 input"):
            if not input_files:
                st.error("请先上传文件。")
            else:
                save_uploaded_files(input_files, INPUT_DIR)
                st.success(f"已保存 {len(input_files)} 个文件到 input")

        st.divider()

        if st.button("开始识别"):
            try:
                results = classify_all(
                    face_app=face_app,
                    threshold=threshold
                )

                if not results:
                    st.warning("input 里面没有可识别的图片或视频。")
                else:
                    st.success("识别完成。")

                    st.dataframe(
                        [
                            {
                                "filename": r["filename"],
                                "label": r["label"],
                                "score": round(r["score"], 4),
                                "source_info": r["source_info"],
                            }
                            for r in results
                        ],
                        use_container_width=True
                    )

            except Exception as e:
                st.error(str(e))

    with tab3:
        st.subheader("分类结果")

        if not OUTPUT_DIR.exists():
            st.warning("还没有 output 目录。")
            return

        label_dirs = [
            p for p in sorted(OUTPUT_DIR.iterdir())
            if p.is_dir()
        ]

        if not label_dirs:
            st.write("还没有分类结果。")
        else:
            for label_dir in label_dirs:
                st.markdown(f"### {label_dir.name}")

                image_paths = [
                    p for p in sorted(label_dir.iterdir())
                    if p.suffix.lower() in IMAGE_EXTS
                ]

                if not image_paths:
                    st.write("无图片结果。")
                    continue

                cols = st.columns(4)

                for idx, image_path in enumerate(image_paths):
                    with cols[idx % 4]:
                        st.image(
                            str(image_path),
                            caption=image_path.name,
                            use_container_width=True
                        )

        st.divider()

        if OUTPUT_DIR.exists() and any(OUTPUT_DIR.rglob("*")):
            zip_path = make_output_zip()

            with open(zip_path, "rb") as f:
                st.download_button(
                    label="下载分类结果 ZIP",
                    data=f,
                    file_name="classified_output.zip",
                    mime="application/zip"
                )
    with tab4:
        st.subheader("已提取的人脸特征库")

        gallery_info = get_gallery_summary()

        if not gallery_info["exists"]:
            st.warning("还没有 gallery.pkl。请先在「建立人脸库」里点击生成人脸库。")
        elif gallery_info["error"]:
            st.error(f"读取 gallery.pkl 失败: {gallery_info['error']}")
        else:
            st.success(f"gallery.pkl 已加载，共 {gallery_info['person_count']} 个人。")

            st.dataframe(
                [
                    {
                        "person_name": row["person_name"],
                        "feature_dim": row["feature_dim"],
                    }
                    for row in gallery_info["rows"]
                ],
                use_container_width=True
            )

            with st.expander("查看 embedding 向量前 8 维"):
                for row in gallery_info["rows"]:
                    st.write(f"**{row['person_name']}**")
                    st.code(row["embedding_preview"])

        st.divider()

        st.subheader("图片级特征缓存")

        cache_info = get_embedding_cache_summary()

        if not cache_info["exists"]:
            st.info("还没有 embedding_cache.pkl。如果你没启用增量缓存，可以忽略这里。")
        elif cache_info["error"]:
            st.error(f"读取 embedding_cache.pkl 失败: {cache_info['error']}")
        else:
            st.success(f"embedding_cache.pkl 已加载，共缓存 {cache_info['total_items']} 张图片记录。")

            if cache_info["rows"]:
                st.dataframe(
                    cache_info["rows"],
                    use_container_width=True
                )
            else:
                st.write("缓存里还没有图片记录。")


if __name__ == "__main__":
    main()
