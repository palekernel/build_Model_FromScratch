# build_Model_FromScratch

本项目旨在从零实现多种主流深度学习模型结构，便于学习和理解其原理与代码实现。各子模块均为独立实现，适合初学者和进阶者参考。

## 目录结构

- `cnn/`：实现基础卷积神经网络（CNN），包含模型定义、训练与测试脚本。
- `Transformer/`：实现 Transformer 结构，含数据处理、模型、训练与测试等。
- `Unet/`：实现经典的 U-Net 图像分割网络。
- `ViT/`：实现视觉Transformer（Vision Transformer, ViT）及其相关模块。

## 各模块简介

### cnn
基础卷积神经网络实现，适用于图像分类等任务。包含：
- `model.py`：CNN模型结构定义。
- `train.py`、`test.py`：训练与测试脚本。

### Transformer
实现了Transformer编码器结构，适合文本等序列建模任务。包含：
- `dataset.py`：数据读取与预处理。
- `model.py`：Transformer模型结构。
- `train.py`、`test.py`：训练与测试脚本。

### Unet
实现U-Net结构，广泛用于医学图像分割等场景。包含：
- `model.py`：U-Net模型结构。

### ViT
实现视觉Transformer及其变体，适用于图像分类等任务。包含：
- `model.py`、`model1.py`：ViT及其变体结构。
- `module/ECA.py`：ECA注意力模块实现。
- `train.py`：训练脚本。

## 许可证

本项目采用 MIT License，详见 LICENSE 文件。
