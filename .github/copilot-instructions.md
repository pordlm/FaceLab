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
```

No test suite, linter, or build pipeline exists.

## Architecture

FaceLab is a Streamlit desktop app for local face recognition. It uses InsightFace (buffalo_l model) through ONNX Runtime for detection and feature extraction, then matches faces via cosine similarity on L2-normalized embeddings.

**Core pipeline:**

```
dataset/{person}/ → [face_engine] → gallery.pkl (per-person mean embeddings)
                                       embedding_cache.pkl (per-image cache)
input/ (query)   → [face_engine] → compare → output/{label}/
```

**Module dependency graph:**

```
config.py          ← no imports, defines all paths/extensions/defaults
face_engine.py     ← config (implicitly via insightface), numpy
image_io.py        ← cv2, numpy
video_utils.py     ← config (VIDEO_EXTS), face_engine (get_largest_face, get_face_area)
gallery_builder.py ← config, face_engine, image_io
classifier.py      ← config, face_engine, gallery_builder, video_utils, image_io
pkl_viewer.py      ← config
self_check.py      ← config, face_engine, gallery_builder, classifier, video_utils, image_io
app.py             ← everything above — the Streamlit UI entry point
```

**Two video strategies exist — do not confuse them:**
- `extract_best_frame_from_video` (video_utils.py): samples ~8 frames from 15%–85% of the video, returns the single frame with the largest face. Used during **classification**.
- `extract_face_frames_from_video` (video_utils.py): samples ~30 frames from 10%–90%, returns up to `max_frames` (default 8) frames with faces sorted by area. Used during **gallery enrollment**.

**Face detection always picks the largest face** (`get_largest_face` in face_engine.py). Multi-face scenes are not handled beyond this.

**Incremental caching** (gallery_builder.py): Each image is keyed by absolute path. Cache validity = same file size + mtime + person_name. Versioned (`CACHE_VERSION = 1`); mismatched version silently resets the cache.

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
