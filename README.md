# FaceLab

FaceLab 是一个基于 InsightFace 和 Streamlit 的本地人脸识别实验项目。项目支持从图片或视频中提取人脸特征，建立本地人脸特征库，并对待识别图片或视频进行身份匹配。

本项目主要用于本地学习、实验和原型验证，不适合作为生产环境门禁系统直接使用。

## 功能特性

- 支持通过图片建立本地人脸库
- 支持通过视频抽帧建立本地人脸库
- 支持图片识别
- 支持视频抽帧识别
- 支持人脸特征缓存，避免重复提取已有图片特征
- 支持在图形化界面中查看已入库人员信息
- 支持查看 `gallery.pkl` 和 `embedding_cache.pkl` 的概要状态
- 支持将识别结果按人员名称输出到不同文件夹
- 支持将分类结果导出为 ZIP 文件

## 技术栈

- Python
- Streamlit
- InsightFace
- ONNX Runtime
- OpenCV
- NumPy
- Pillow
- scikit-learn

## 项目结构

```text
facelab/
  app.py
  config.py
  face_engine.py
  gallery_builder.py
  classifier.py
  video_utils.py
  image_io.py
  pkl_viewer.py
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

## 主要文件说明

| 文件 | 说明 |
|---|---|
| `app.py` | Streamlit 图形化界面入口 |
| `config.py` | 路径、文件类型、默认参数配置 |
| `face_engine.py` | InsightFace 模型加载、人脸检测、特征提取、相似度计算 |
| `gallery_builder.py` | 从 `dataset/` 中提取人脸特征，生成 `gallery.pkl` |
| `classifier.py` | 对 `input/` 中的图片或视频进行识别分类 |
| `video_utils.py` | 视频抽帧相关逻辑 |
| `image_io.py` | 图片读取与保存工具 |
| `pkl_viewer.py` | 读取并展示特征库和缓存文件的概要信息 |
| `start.ps1` | Windows PowerShell 启动脚本 |

## 安装环境

建议使用独立的 Python 环境，避免污染系统环境或 Anaconda 的 base 环境。

推荐 Python 版本：`3.10`

### 使用 Conda 创建环境

```bash
conda create -n facelab python=3.10 -y
conda activate facelab
```

### 安装 CPU 版本依赖

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

如果上传视频，程序会自动抽取包含人脸的帧，并将抽帧结果保存为图片。

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
- `embedding_cache.pkl` 保存每张已处理图片的人脸特征缓存

### 2. 识别图片或视频

进入“识别图片/视频”页面，上传待识别的图片或视频。

如果是图片，程序会直接进行人脸识别。

如果是视频，程序会自动抽取一帧包含人脸的画面进行识别。

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
    test_video_frame.jpg
  unknown/
    unknown_photo.jpg
```

### 3. 查看特征库状态

进入“特征库状态”页面，可以查看：

- 已入库人员名称
- 每个人的特征维度
- 图片级缓存数量
- 缓存成功和失败的图片数量
- embedding 向量的前几维预览

完整 embedding 向量通常为高维数值，不建议在界面中完整展示。

## 阈值说明

项目使用余弦相似度进行人脸匹配。默认阈值为：

```text
0.45
```

一般来说：

- 阈值越低，越容易识别为已知人员，但误识别风险更高
- 阈值越高，识别更严格，陌生人更容易被归为 `unknown`

可以根据自己的数据在界面中调整阈值。

## GPU 依赖安装

默认情况下，项目使用 CPU 版本的 `onnxruntime`。

如果需要使用 NVIDIA GPU 加速，需要将 `onnxruntime` 替换为 `onnxruntime-gpu`。

### 1. 卸载 CPU 版 ONNX Runtime

```bash
pip uninstall onnxruntime -y
```

### 2. 安装 GPU 版 ONNX Runtime

```bash
pip install onnxruntime-gpu
```

### 3. 验证 ONNX Runtime 可用的执行设备

运行：

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

如果输出中包含：

```text
CUDAExecutionProvider
```

说明 ONNX Runtime 已识别到 CUDA GPU。

例如：

```text
['CUDAExecutionProvider', 'CPUExecutionProvider']
```

如果只看到：

```text
['CPUExecutionProvider']
```

说明当前环境没有成功启用 GPU 推理。

### 4. 在程序中启用 GPU

如果项目界面中有“使用 GPU”的选项，勾选后会使用 GPU 推理。

对应代码逻辑通常位于 `face_engine.py` 中：

```python
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
```

CPU 模式使用：

```python
providers = ["CPUExecutionProvider"]
ctx_id = -1
```

GPU 模式使用：

```python
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
ctx_id = 0
```

### 5. GPU 环境注意事项

使用 GPU 版本时，需要确保本机具备：

- NVIDIA 显卡
- 可用的 NVIDIA 驱动
- 与 `onnxruntime-gpu` 版本匹配的 CUDA / cuDNN 环境

如果 GPU 依赖配置不正确，程序可能会自动回退到 CPU，或者在启动时出现动态库加载错误。

如果不确定 GPU 环境是否正确，建议先使用 CPU 版本运行项目。

## PyTorch 说明

本项目的人脸推理主要依赖 InsightFace 和 ONNX Runtime，并不需要主动使用 PyTorch 进行模型推理。

某些依赖库可能会间接导入 PyTorch。如果在 Windows 环境下遇到类似：

```text
Error loading "...torch\lib\fbgemm.dll" or one of its dependencies
```

通常是 PyTorch 或系统运行库依赖存在问题。

建议处理方式：

1. 使用独立 Conda 环境，不要直接使用 Anaconda base 环境
2. 重新安装 PyTorch
3. 安装或修复 Microsoft Visual C++ Redistributable
4. 优先确认以下命令可以正常运行：

```bash
python -c "import torch; print(torch.__version__)"
python -c "from insightface.app import FaceAnalysis; print('InsightFace OK')"
```

如果项目只使用 CPU，且环境中 PyTorch 导入失败，可以优先尝试重新安装相关依赖，或使用干净环境重新部署。

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
```

本仓库的 `.gitignore` 应默认排除这些文件。

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

# Logs
*.log
```

## 注意事项

- 注册人脸库的视频建议只包含单人
- 上传人脸库的图片应尽量清晰、正脸、光线正常
- 如果视频中存在多人，程序默认会优先选取画面中较大的人脸
- 本项目不包含完整活体检测能力
- 本项目不建议直接用于真实门禁、考勤或安防系统
- `gallery.pkl` 和 `embedding_cache.pkl` 虽然不是原始图片，但仍然属于人脸生物特征数据，不应上传到公开仓库

## 许可证

本项目采用 GNU General Public License v3.0，详见 [LICENSE](./LICENSE)。
