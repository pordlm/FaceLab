# Copilot Instructions for FaceLab

## Run Commands

```bash
# Launch the Streamlit app
streamlit run app.py

# Quick environment check
python -c "from insightface.app import FaceAnalysis; print('InsightFace OK')"
python -c "import cv2, numpy; print('OpenCV + NumPy OK')"

# Check ONNX Runtime providers (GPU: CUDAExecutionProvider, CPU only: CPUExecutionProvider)
python -c "import onnxruntime as ort; print(ort.get_available_providers())"

# Syntax check before committing (no full test suite exists)
python -m py_compile app.py classifier.py self_check.py face_engine.py gallery_builder.py video_utils.py image_io.py pkl_viewer.py
```

No test suite, linter, or build pipeline exists. `py_compile` is the closest thing to validation.

## Architecture

FaceLab is a Streamlit desktop app for local face recognition. It uses InsightFace (buffalo_l model) through ONNX Runtime for detection and feature extraction, then matches faces via cosine similarity on L2-normalized embeddings.

**Core pipeline:**

```
dataset/{person}/ → [face_engine] → gallery.pkl (per-person mean embeddings)
                                       embedding_cache.pkl (per-image cache)
input/ (query)   → [face_engine] → compare → output/{label}/
output/           → [self_check]  → finds read_failed + missing → reclassify → merge results
```

**Module dependency graph:**

```
config.py          ← no imports; defines all paths/extensions/defaults, ensure_dirs()
face_engine.py     ← config (implicitly via insightface model path), numpy
image_io.py        ← cv2, numpy
video_utils.py     ← config (VIDEO_EXTS), face_engine (get_largest_face, get_face_area)
gallery_builder.py ← config, face_engine, image_io
classifier.py      ← config, face_engine, gallery_builder, video_utils, image_io
pkl_viewer.py      ← config (reads pickle files for UI display)
self_check.py      ← config, face_engine, gallery_builder, classifier, video_utils, image_io
app.py             ← everything above — the Streamlit UI entry point
```

**Two video strategies exist — do not confuse them:**
- `extract_best_frame_from_video` (video_utils.py): samples ~8 frames from 15%–85% of the video, returns the single frame with the largest face. Used during **classification**.
- `extract_face_frames_from_video` (video_utils.py): samples ~30 frames from 10%–90%, returns up to `max_frames` (default 8) frames with faces sorted by area. Used during **gallery enrollment**.

**Face detection always picks the largest face** (`get_largest_face` in face_engine.py). Multi-face scenes are not handled beyond this.

**Incremental caching** (gallery_builder.py): Each image is keyed by absolute path. Cache validity = same file size + mtime + person_name. Versioned (`CACHE_VERSION = 1`); mismatched version silently resets the cache.

**Self-check flow** (self_check.py): After classification, `run_self_check()` reads `classification_results.json` as the authoritative source. `find_problems()` matches `input/` files against JSON entries by filename and finds three problem types: (1) input files with no corresponding JSON record, (2) entries where `decision == "read_failed"`, and (3) entries where non-empty `output_path` or `frame_path` points to a file that doesn't exist on disk. Problem files are reclassified via `reclassify_files()`, and `_merge_and_save_results()` upserts the new results into `classification_results.json` by filename key (existing entries not in the reclassify set are left untouched).

**⚠️ Duplicate classification logic:** `classifier.classify_all()` and `self_check.reclassify_files()` contain nearly identical logic for image loading, face extraction, file copying, and output path construction. Any behavioral change to one function (e.g., how `output_path`/`frame_path` are set, how `no_face` videos are handled) **must be mirrored in the other**. They share helper functions (`load_image_or_video_frame`, `predict_person`, `is_supported_file`) but the per-file result construction is duplicated. If you refactor one, refactor both.

**Streamlit patterns:**
- `@st.cache_resource` wraps `create_face_app()` in `app.py` — the face app is created once and reused across reruns. GPU toggle changes invalidate the cache.
- Uploader widgets use `st.session_state` counter keys (`known_uploader_key`, `input_uploader_key`) so that clearing resets the widget state via `st.rerun()`.
- `translate_decision()` maps internal decision labels to Chinese display strings: `no_face`→"未检测到人脸", `read_failed`→"文件读取失败", `face_detected_unknown`→"检测到人脸，未匹配到已知人员", `face_detected_known`→"检测到人脸，已匹配到已知人员".

**`classification_results.json` schema:** Each entry is a dict with these keys:
| Key | Type | Description |
|-----|------|-------------|
| `filename` | string | Original input filename |
| `label` | string | Person name, `"unknown"`, `"no_face"`, or `"read_failed"` |
| `score` | float | Cosine similarity (0.0 for non-matches) |
| `decision` | string | One of the four decision labels above |
| `source_info` | string | Human-readable processing note |
| `output_path` | string | Path to the copied file in `output/` (image: copy of original; video with face: copy of original video; video no_face: points to the frame JPEG; read_failed: empty) |
| `frame_path` | string | Path to the extracted frame JPEG (video only; always `""` for images and `no_face` videos) |

`load_classification_results()` returns `list[dict] | None` — `None` means no prior classification has been performed yet.

## Key Conventions

- **All paths are `pathlib.Path`.** Never use `os.path` or string paths.
- **Image I/O uses `np.fromfile` + `cv2.imdecode`/`cv2.imencode`**, not `cv2.imread`/`cv2.imwrite`. This is intentional — `cv2.imread` fails on non-ASCII Windows paths (e.g., Chinese characters). Always follow this pattern.
- **Embeddings are L2-normalized** after extraction and after averaging. Cosine similarity becomes a simple dot product (`np.dot`). Do not skip normalization.
- **Gallery key type:** `gallery.pkl` is a `dict[str, np.ndarray]` mapping person name → mean embedding. `embedding_cache.pkl` is `{"version": int, "items": {str_path: {...}}}`.
- **Decision labels** for classification results: `read_failed`, `no_face`, `face_detected_unknown`, `face_detected_known`. The UI translates these to Chinese. Keep these exact string values.
- **config.py is the single source of truth** for directory paths, file extensions, and defaults. Import from it; never hardcode paths like `"dataset"` or `".jpg"` as strings elsewhere.
- **`classify_all()` wipes and recreates `output/`** on every run. If you need to preserve results between runs, modify this behavior explicitly.
- **Video frame extraction skips head/tail** to avoid title cards and credits: 10%–90% for enrollment, 15%–85% for classification.
- **GPU support** is toggled via the `use_gpu` checkbox in the Streamlit sidebar. `face_engine.create_face_app(use_gpu)` sets ONNX Runtime providers accordingly. The face app is cached with `@st.cache_resource` in `app.py`.
- **Temp file pattern for video enrollment:** `save_known_files_for_person()` in `app.py` writes video uploads to a `_temp_` prefixed file, extracts face frames from it, then deletes the temp file. Image uploads use the same temp-then-rename pattern. Do not skip the temp step — it avoids filename collisions.
- **Video classification output has two paths:** For videos with a detected face, `output_path` points to the copied original video file and `frame_path` points to the extracted frame JPEG. For `no_face` videos, `output_path` points to the extracted frame JPEG and `frame_path` is `""` (no original video is copied). For images, `frame_path` is always `""`.

- **`start.ps1`** is a convenience launcher. It should use `python` (from PATH) rather than a hardcoded absolute path like `E:\Anaconda3\envs\facelab\python.exe`, which is machine-specific.

- **`scikit-learn`** is listed in `requirements.txt` but is **not imported by any module**. It can be safely removed from dependencies.
