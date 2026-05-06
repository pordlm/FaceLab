# FaceLab

FaceLab 是一个基于 **InsightFace + ONNX Runtime + Streamlit** 的本地人脸识别实验项目。项目支持从图片或视频中提取人脸特征，建立本地人脸特征库，并对待识别图片或视频进行身份匹配。

> 本项目主要用于本地学习、实验和原型验证，不适合作为生产环境门禁、考勤或安防系统直接使用。

---

## 功能特性

- 支持通过图片建立本地人脸库
- 支持通过视频抽帧建立本地人脸库
- 支持图片识别
- 支持视频抽帧识别
- 支持先判断图片或视频帧中是否存在人脸，再进行身份识别
- 支持将识别结果区分为：
  - 已知人员
  - `unknown`
  - `no_face`
  - `read_failed`
- 支持视频识别结果同时输出：
  - 原始视频文件
  - 用于识别和预览的抽帧图片
- 支持人脸特征缓存，避免重复提取已有注册图片特征
- 支持在图形化界面中查看已入库人员信息
- 支持查看 `gallery.pkl` 和 `embedding_cache.pkl` 的概要状态
- 支持分类结果自检，检查读取失败、输出缺失等问题
- 支持对部分问题文件进行重新分类
- 支持将识别结果按人员名称输出到不同文件夹
- 支持将分类结果导出为 ZIP 文件
- 支持在界面中清空 `input/` 上传文件和 `output/` 识别结果
- 支持 GPU 模式下使用 `CUDAExecutionProvider`
- 支持可选 profiling 日志，用于分析 detection、recognition、视频抽帧、文件 IO 等耗时
- 已对 InsightFace 调用路径做轻量优化：
  - 只加载 FaceLab 实际使用的 `detection` 和 `recognition` 模型
  - 视频抽帧阶段仅执行 detection，不再做无用 recognition

---

## 技术栈

- Python
- Streamlit
- InsightFace
- ONNX Runtime / ONNX Runtime GPU
- OpenCV
- NumPy
- Pillow
- scikit-learn

---

## 项目结构

```text
facelab/
  app.py
  config.py
  face_engine.py
  gallery_builder.py
  classifier.py
  self_check.py
  video_utils.py
  image_io.py
  pkl_viewer.py
  profiler.py
  start.ps1
  requirements.txt
  README.md
  LICENSE
  .gitignore

  dataset/
    .gitkeep
  input/
    .gitkeep
  output/
    .gitkeep
```

---

## 主要文件说明

| 文件 | 说明 |
|---|---|
| `app.py` | Streamlit 图形化界面入口 |
| `config.py` | 路径、文件类型、默认参数配置 |
| `face_engine.py` | InsightFace 模型加载、人脸检测、特征提取、相似度计算 |
| `gallery_builder.py` | 从 `dataset/` 中提取人脸特征，生成 `gallery.pkl` 和 `embedding_cache.pkl` |
| `classifier.py` | 对 `input/` 中的图片或视频进行识别分类，并生成分类结果 |
| `self_check.py` | 对识别结果进行自检，检查读取失败和输出缺失等问题 |
| `video_utils.py` | 视频抽帧相关逻辑 |
| `image_io.py` | 图片读取与保存工具 |
| `pkl_viewer.py` | 读取并展示特征库和缓存文件的概要信息 |
| `profiler.py` | 可选 profiling 工具模块，受 `FACELAB_PROFILE=1` 控制 |
| `start.ps1` | Windows PowerShell 启动脚本 |

---

## 安装环境

建议使用独立 Python 环境，避免污染系统环境或 Anaconda 的 `base` 环境。

推荐 Python 版本：

```text
Python 3.10 或 3.11
```

### 使用 Conda 创建环境

```bash
conda create -n facelab python=3.11 -y
conda activate facelab
```

也可以使用 Python 3.10：

```bash
conda create -n facelab python=3.10 -y
conda activate facelab
```

---

## 安装 CPU 版本依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 示例：

```text
streamlit
insightface
onnxruntime
opencv-python
numpy
pillow
scikit-learn
```

CPU 版本适合普通电脑运行，不需要 NVIDIA 显卡。

---

## 启动项目

### 方式一：命令行启动

```bash
streamlit run app.py
```

### 方式二：Windows PowerShell 启动

```powershell
.\start.ps1
```

如果 PowerShell 阻止脚本运行，可以临时使用：

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

推荐的 `start.ps1` 内容：

```powershell
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

python -m streamlit run app.py
```

---

## 使用流程

### 1. 建立人脸库

进入界面中的“建立人脸库”页面。

为每个人输入一个名称，例如：

```text
zhangsan
lisi
```

然后上传该人员的图片或视频。

如果上传图片，程序会直接保存到对应人员目录。

如果上传视频，程序会自动抽取包含人脸的帧，并将抽帧结果保存为图片。注册人脸库时的视频建议只包含单人，避免把其他人的人脸抽入该人员目录。

上传完成后，点击：

```text
生成人脸库 gallery.pkl
```

程序会从 `dataset/` 中提取人脸特征，并生成：

```text
gallery.pkl
embedding_cache.pkl
```

其中：

- `gallery.pkl` 保存每个人的平均人脸特征
- `embedding_cache.pkl` 保存每张已处理注册图片的人脸特征缓存

后续再次建立人脸库时，未变化的注册图片会优先复用缓存；新增或修改过的图片才会重新提取特征。

---

### 2. 识别图片或视频

进入“识别图片/视频”页面，上传待识别的图片或视频。

如果是图片，程序会直接进行人脸检测和身份识别。

如果是视频，程序会自动抽取包含人脸的画面进行识别。识别完成后，程序会同时保存：

- 原始视频文件
- 用于识别和预览的抽帧图片

识别完成后，结果会输出到：

```text
output/
```

示例：

```text
output/
  zhangsan/
    test_photo.jpg
  lisi/
    test_video.mp4
    test_video_frame.jpg
  unknown/
    unknown_photo.jpg
  no_face/
    landscape.jpg
  read_failed/
```

视频识别结果中：

- `output_path` 指向原始视频文件
- `frame_path` 指向用于识别和预览的抽帧图片
- 图片识别结果的 `frame_path` 通常为空字符串

在“识别图片/视频”页面可以清空 `input/` 上传文件；在“查看结果”页面可以清空 `output/` 识别结果。

---

### 3. 查看结果

进入“查看结果”页面，可以查看按识别结果分类后的图片和视频。

识别结果说明：

| 结果 | 含义 |
|---|---|
| 已知人员名称 | 检测到人脸，并成功匹配到人脸库中的人员 |
| `unknown` | 检测到人脸，但未匹配到已知人员 |
| `no_face` | 未检测到人脸 |
| `read_failed` | 文件读取失败 |

---

### 4. 查看特征库状态

进入“特征库状态”页面，可以查看：

- 已入库人员名称
- 每个人的特征维度
- 图片级缓存数量
- 缓存成功和失败的图片数量
- embedding 向量的前几维预览

完整 embedding 向量通常为高维数值，不建议在界面中完整展示。

---

### 5. 分类结果自检

进入“自检”页面后，可以运行分类结果自检。

自检主要检查：

- `classification_results.json` 中是否存在 `decision` 为 `read_failed` 的条目
- `classification_results.json` 中 `output_path` 或 `frame_path` 指向的文件是否存在
- `input/` 目录下的文件在 `classification_results.json` 中是否有对应记录
- 是否可以对上述问题文件重新分类

如果问题文件重分类成功，程序会将结果写入对应的 `output/` 子目录，并同步更新：

```text
output/classification_results.json
```

这样可以避免下次自检继续读取旧状态。

---

## 输出文件说明

识别完成后，`output/` 中可能包含：

```text
output/
  classification_results.json
  zhangsan/
  lisi/
  unknown/
  no_face/
  read_failed/
```

其中：

- `classification_results.json` 保存本次识别的结构化结果
- 已知人员目录保存匹配成功的文件
- `unknown/` 保存检测到人脸但未匹配到已知人员的文件
- `no_face/` 保存未检测到人脸的文件
- `read_failed/` 用于保存或标记读取失败的文件结果

导出 ZIP 时，程序会将当前 `output/` 中的分类结果打包为：

```text
classified_output.zip
```

该文件属于运行结果，不应提交到 Git 仓库。

---

## 阈值说明

项目使用余弦相似度进行人脸匹配。默认阈值为：

```text
0.45
```

一般来说：

- 阈值越低，越容易识别为已知人员，但误识别风险更高
- 阈值越高，识别更严格，陌生人更容易被归为 `unknown`

可以根据自己的数据在界面中调整阈值。

---

## InsightFace 模型加载优化

FaceLab 使用 InsightFace 的 `buffalo_l` 模型包，但项目实际只需要：

- `detection`
- `recognition`

因此当前 `face_engine.py` 中创建 `FaceAnalysis` 时使用：

```python
app = FaceAnalysis(
    name="buffalo_l",
    providers=providers,
    allowed_modules=["detection", "recognition"],
)
```

这样会跳过 FaceLab 不使用的模型：

- `landmark_3d_68`
- `landmark_2d_106`
- `genderage`

该优化不改变：

- 模型包：仍使用 `buffalo_l`
- `det_size`
- threshold
- providers 逻辑
- recognition embedding 逻辑
- no_face / read_failed 语义

InsightFace 的 ArcFace recognition 对齐使用的是 detection 模型输出的 `kps`，不依赖额外的 landmark 模型。因此跳过上述未使用模型不会降低 FaceLab 当前识别逻辑的准确率。

---

## 视频抽帧优化

视频抽帧阶段只需要判断：

```text
当前帧是否有人脸，以及最大人脸面积是多少
```

因此当前 `video_utils.py` 在视频采样阶段直接调用：

```python
bboxes, _ = face_app.det_model.detect(frame)
```

而不是：

```python
face_app.get(frame)
```

这样视频采样阶段只执行 detection，不再执行无用的 recognition。

正式分类时，程序仍会对最终选出的 `best_frame` 调用完整的 embedding 提取流程：

```text
best_frame -> extract_embedding -> face_app.get(best_frame)
```

因此该优化不改变识别结果，只减少视频采样阶段的重复推理。

---

## GPU 依赖安装与 CUDA/cuDNN 支持

默认情况下，项目可以使用 CPU 版本 `onnxruntime` 运行。

如果需要使用 NVIDIA GPU 加速，需要使用 `onnxruntime-gpu`，并确保 CUDA / cuDNN 依赖可以被 Windows 进程找到。

### 1. 卸载 CPU 版 ONNX Runtime

```bash
pip uninstall onnxruntime -y
```

### 2. 安装 GPU 版 ONNX Runtime

```bash
pip install onnxruntime-gpu
```

### 3. 安装或提供 CUDA / cuDNN 依赖

以 `onnxruntime-gpu 1.25.1` 为例，实测其 GPU build 使用 CUDA 12.x，并需要 cuDNN 9 相关 DLL。

Windows 下如果缺少 cuDNN 9，可能会出现类似错误：

```text
Error loading "...onnxruntime_providers_cuda.dll" which depends on "cudnn64_9.dll" which is missing.
Failed to create CUDAExecutionProvider.
Require cuDNN 9.* and CUDA 12.*.
```

可选处理方式：

#### 方式 A：安装 NVIDIA pip 依赖包

```bash
pip install nvidia-cudnn-cu12
pip install nvidia-cublas-cu12
pip install nvidia-cuda-nvrtc-cu12
```

实际需要哪些包取决于当前 `onnxruntime-gpu` 版本和本机环境。如果已经安装了完整 CUDA / cuDNN，也可以不使用 pip 包方式。

#### 方式 B：手动安装 CUDA / cuDNN

安装与 `onnxruntime-gpu` 版本匹配的：

- NVIDIA Driver
- CUDA Toolkit
- cuDNN

并确保相关 `bin` 目录在 `PATH` 中。

### 4. Windows PowerShell 中临时加入 NVIDIA DLL 路径

如果通过 pip 安装了 NVIDIA CUDA/cuDNN 相关包，可以在启动前把其 `bin` 目录加入当前 PowerShell 进程的 `PATH`：

```powershell
$nvRoot = "E:\Anaconda3\envs\facelab\Lib\site-packages\nvidia"
$nvBins = Get-ChildItem $nvRoot -Recurse -Directory -Filter bin | Select-Object -ExpandProperty FullName
$env:PATH = ($nvBins -join ";") + ";" + $env:PATH
```

然后启动：

```powershell
E:\Anaconda3\envs\facelab\python.exe -m streamlit run app.py
```

请根据自己的环境修改 Python 路径和 Conda 环境路径。

### 5. 不要只看 get_available_providers()

下面命令只能说明 ONNX Runtime 环境“看得到” CUDA provider：

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

即使输出包含：

```text
CUDAExecutionProvider
```

也不代表具体模型 session 已经真的用上 CUDA。

更可靠的验证方式是检查具体 ONNX session：

```bash
python -c "from pathlib import Path; import onnxruntime as ort; model=Path.home()/'.insightface'/'models'/'buffalo_l'/'det_10g.onnx'; sess=ort.InferenceSession(str(model), providers=['CUDAExecutionProvider','CPUExecutionProvider']); print(sess.get_providers()); print(sess.get_provider_options())"
```

理想输出应包含：

```text
['CUDAExecutionProvider', 'CPUExecutionProvider']
```

如果实际输出只有：

```text
['CPUExecutionProvider']
```

说明 CUDAExecutionProvider 创建失败，ONNX Runtime 已回退到 CPU。

### 6. 在程序中启用 GPU

在界面中勾选“使用 GPU”后，FaceLab 会使用：

```python
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
ctx_id = 0
```

CPU 模式使用：

```python
providers = ["CPUExecutionProvider"]
ctx_id = -1
```

如果 GPU 依赖不完整，即使选择 GPU，ONNX Runtime 也可能自动回退到 CPU。此时需要检查 CUDA / cuDNN DLL 是否可被加载。

---

## Profiling 性能分析

FaceLab 提供可选 profiling 日志，用于分析各阶段耗时和 ONNX Runtime provider 状态。

默认关闭。

### 开启 profiling

PowerShell：

```powershell
$env:FACELAB_PROFILE="1"
python -m streamlit run app.py
```

### 关闭 profiling

PowerShell：

```powershell
Remove-Item Env:\FACELAB_PROFILE -ErrorAction SilentlyContinue
```

### 日志位置

开启后会生成：

```text
profile_logs/profile.jsonl
```

该文件为 JSON Lines 格式，每行一个事件，可能包含：

- `session.created`
- `retinaface.session_run`
- `arcface.session_run`
- `face_analysis.get`
- `video.sample_detect`
- `classify_all`
- `ui.classify`

示例查询：

```powershell
Get-Content .\profile_logs\profile.jsonl |
  Select-String '"providers_actual"'
```

查看关键耗时：

```powershell
Get-Content .\profile_logs\profile.jsonl -Tail 100 |
  Select-String 'retinaface.session_run|arcface.session_run|face_analysis.get|classify_all|ui.classify'
```

### profiling 的用途

Profiling 不是优化本身，而是用来回答：

- 哪一步最慢
- detection / recognition 各自耗时是多少
- 视频抽帧是否慢
- 是否真正使用了 `CUDAExecutionProvider`
- 是否发生了 CPU fallback

---

## PyTorch 说明

本项目的人脸推理主要依赖 InsightFace 和 ONNX Runtime，并不需要主动使用 PyTorch 进行模型推理。

某些依赖库可能会间接导入 PyTorch。如果在 Windows 环境下遇到类似：

```text
Error loading "...torch\lib\fbgemm.dll" or one of its dependencies
```

通常是 PyTorch 或系统运行库依赖存在问题。

建议处理方式：

1. 使用独立 Conda 环境，不要直接使用 Anaconda `base` 环境
2. 重新安装 PyTorch
3. 安装或修复 Microsoft Visual C++ Redistributable
4. 优先确认以下命令可以正常运行：

```bash
python -c "import torch; print(torch.__version__)"
python -c "from insightface.app import FaceAnalysis; print('InsightFace OK')"
```

如果项目只使用 CPU，且环境中 PyTorch 导入失败，可以优先尝试重新安装相关依赖，或使用干净环境重新部署。

---

## 开发与检查

提交代码前建议运行：

```bash
python -m py_compile app.py classifier.py self_check.py
```

如果修改了其他模块，也可以一并检查：

```bash
python -m py_compile app.py classifier.py self_check.py config.py face_engine.py gallery_builder.py video_utils.py image_io.py pkl_viewer.py profiler.py
```

建议提交前确认以下内容没有被 Git 跟踪：

```text
dataset/
input/
output/
gallery.pkl
embedding_cache.pkl
classified_output.zip
profile_logs/
__pycache__/
```

---

## 数据与隐私说明

本项目仅用于本地学习和实验。人脸图片、视频和人脸特征均属于敏感生物识别数据。

请不要将以下内容上传到公开仓库：

```text
dataset/
input/
output/
gallery.pkl
embedding_cache.pkl
classified_output.zip
profile_logs/
```

本仓库的 `.gitignore` 应默认排除这些文件。

---

## 推荐的 .gitignore

```gitignore
# Python cache
__pycache__/
*.py[cod]
*$py.class

# Virtual environments
.venv/
venv/
env/
ENV/
*.egg-info/

# IDE / editor
.vscode/
.idea/

# OS files
.DS_Store
Thumbs.db

# Streamlit local secrets
.streamlit/secrets.toml

# Local biometric data and runtime data
dataset/*
input/*
output/*

# Keep folder structure
!dataset/.gitkeep
!input/.gitkeep
!output/.gitkeep

# Generated biometric feature files
gallery.pkl
embedding_cache.pkl
*.pkl

# Exported results
classified_output.zip
*.zip

# Local media files
*.jpg
*.jpeg
*.png
*.webp
*.mp4
*.avi
*.mov
*.mkv

# Logs and profiling
*.log
profile_logs/
*.jsonl
```

---

## 注意事项

- 注册人脸库的视频建议只包含单人
- 上传人脸库的图片应尽量清晰、正脸、光线正常
- 如果注册视频中存在多人，程序默认会优先选取画面中较大的人脸
- 视频识别结果会保存原始视频和抽帧图片，输出文件体积可能较大
- 本项目不包含完整活体检测能力
- 本项目不建议直接用于真实门禁、考勤或安防系统
- `gallery.pkl` 和 `embedding_cache.pkl` 虽然不是原始图片，但仍然属于人脸生物特征数据，不应上传到公开仓库
- `profile_logs/profile.jsonl` 可能包含本地文件路径、文件名、耗时和识别标签，也不建议上传到公开仓库

---

## 许可证

本项目采用 GNU General Public License v3.0，详见 [LICENSE](./LICENSE)。
