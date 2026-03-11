# Machine_box_YoloV11

## 项目描述

这是一个基于 YOLOv11 的药盒智能检测系统，能够自动检测药盒中的目标区域，并使用 PaddleOCR 进行文字识别。系统提供了一个简约美观的 PyQt5 图形界面，支持图片选择、目标检测和文字识别功能。

## 功能特性

- **目标检测**：使用 YOLOv11 模型检测药盒中的特定区域（如药品名称、剂量等）。
- **文字识别**：集成 PaddleOCR，支持中文文字识别，并提供置信度信息。
- **可视化结果**：检测结果以标注框的形式可视化显示。
- **图形界面**：基于 PyQt5 的现代简约界面，支持全屏、拖拽调整大小。
- **异常处理**：OCR 和检测过程中包含完善的异常处理机制。
- **相对路径**：所有数据路径使用相对路径，便于项目移植。

## 安装依赖

确保您的环境已安装 Python 3.8+，然后运行以下命令安装依赖：

```bash
pip install ultralytics paddlepaddle paddleocr PyQt5 opencv-python
```

如果您的系统支持 GPU，可以安装 GPU 版本的 PaddlePaddle 以加速 OCR：

```bash
pip install paddlepaddle-gpu
```

## 使用方法

### 1. 运行主程序

```bash
python main.py
```

这将启动图形界面。

### 2. 操作步骤

1. **选择图片**：点击“选择图片”按钮，选择一张药盒图片。
2. **开始检测**：点击“开始检测”按钮，系统将使用 YOLO 模型检测图片中的目标区域，并显示标注框。
3. **文字识别**：点击“文字识别”按钮，系统将对检测到的区域进行 OCR，并显示识别结果。

### 3. 命令行预测

您也可以使用命令行进行预测：

```bash
python Predict.py
```

## 训练模型

如果需要重新训练模型，请运行：

```bash
python train.py
```

训练脚本包含了自适应学习率、早停、数据增强等优化配置。训练结果将保存在 `runs/train` 目录中。

## 数据集

数据集位于 `data/medicine-box.v7i.yolov11/` 目录，包含训练、验证和测试集。数据集配置文件为 `data.yaml`，包含 22 个类别（药品编号）。

数据集来源：Roboflow Universe - medicine-box dataset v7。

## 项目结构

- `main.py`：主程序，PyQt5 图形界面。
- `Predict.py`：YOLO 预测脚本。
- `OCR.py`：PaddleOCR 文字识别脚本。
- `train.py`：模型训练脚本。
- `data/`：数据集目录。
- `yolo11n-medicine-box2/`：训练好的模型和结果。
- `runs/`：训练和预测结果目录。

## 注意事项

- 模型推理优先使用 ultralytics 官方 API。
- 推理结果必须提供可视化（标注框 + 标签 + 置信度）。
- OCR 代码包含异常处理（图片读取失败、识别结果为空等）。
- 所有代码注释使用中文。

## 许可证

本项目采用 MIT 许可证。
