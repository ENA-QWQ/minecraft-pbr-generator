# MCPG - Minecraft PBR Generator

[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE) [![Java](https://img.shields.io/badge/Java-8%2B-red.svg)](https://adoptium.net/) [![Maven](https://img.shields.io/badge/Build-Maven-C71A36.svg)](https://maven.apache.org/)

## 描述

本项目是一个概念验证工具，利用小型神经网络从 MC 纹理自动生成 LabPBR 贴图。

神经网络只从纹理预测高度图。法线贴图通过 Sobel 风格的差分算法从高度图推导，粗糙度和金属度直接从原图的灰度和红色通道映射生成。（以后会改进高光贴图的生成算法）

项目分为三个独立模块，分别负责数据准备、模型训练和推理生成，可以根据需要单独使用。

不追求生产级精度或性能。模型结构轻量，仅在有限数据集上测试，输出效果可能存在瑕疵。欢迎用于学习、实验或二次开发。

---

This project is a proof-of-concept tool that uses a small neural network to automatically generate LabPBR maps from MC textures.

The neural network predicts only the height map from the texture. The normal map is derived from the height map via a Sobel‑style difference algorithm, while roughness and metalness are mapped directly from the grayscale and red channel of the original image. (The specular map generation will be improved in the future.)

The project is divided into three independent modules responsible for data preparation, model training, and inference generation, which can be used separately as needed.

It does not aim for production‑grade accuracy or performance. The model architecture is lightweight and has only been tested on a limited dataset, so the output may have imperfections. It is welcome for learning, experimentation, or further development.

---

## 使用说明

### 中文

#### 模块一：数据集构建

**JAR：** `labpbr-dataset-builder-1.0.0.jar`  
**运行：**
```bash
java -jar labpbr-dataset-builder-1.0.0.jar [选项]
```
**参数：**

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--inputDir` | 资源包目录 | `./resourcepacks` |
| `--outputDir` | 输出目录 | `./dataset` |
| `--maxSamples` | 最大采样数 | `20000000` |
| `--seed` | 随机种子 | 系统时间 |

---

#### 模块二：训练

**JAR：** `tiny-mlp-training-1.0.0.jar`  
**运行：**
```bash
java -jar tiny-mlp-training-1.0.0.jar [选项]
```
**参数：**

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--data` | 数据文件 | `train_data.bin` |
| `--labels` | 标签文件 | `train_labels.bin` |
| `--output` | 模型输出路径 | `height_model.ser` |
| `--batch-size` | 批量大小 | `64` |
| `--epochs` | 最大轮数 | `50` |
| `--patience` | 早停耐心 | `8` |
| `--lr` | 初始学习率 | `0.01` |
| `--lr-decay` | 衰减因子 | `0.9` |
| `--lr-step` | 衰减步长 | `10` |
| `--seed` | 随机种子 | `42` |
| `--total-samples` | 总样本数 | `2000000` |
| `--train-size` | 训练集样本数 | `1000000` |
| `--val-size` | 验证集样本数 | `20000` |
| `--help` | 显示帮助 | — |

---

#### 模块三：推理

**JAR：** `pbr-inference-1.0.0.jar`  
**运行：**
```bash
java -jar pbr-inference-1.0.0.jar --input <输入图片> [选项]
```
**参数：**

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--input` | 输入图片路径 | 必填 |
| `--outputDir` | 输出目录 | `./output` |
| `--model` | 模型路径 | `height_model.ser` |
| `--strength` | 法线强度 | `6.0` |
| `--pixelate` | 像素化开关 | `false` |
| `--smoothness` | 基础光滑度 | `0.2` |
| `--metallic` | 基础金属度 | `0.0` |
| `--invert-height` | 反转高度 | `true` |
| `--invert-normal-y` | 反转法线Y | `false` |
| `--height-strength` | 高度对比度 | `1.2` |
| `--height-min` | 高度最小值 | `0.2` |
| `--height-max` | 高度最大值 | `1.0` |
| `--height-smooth` | 模糊半径 | `2` |
| `--norm-percentile` | 百分位截断 | `2.0` |
| `--help` | 显示帮助 | — |

---

### English

#### Module 1: Dataset Builder

**JAR:** `labpbr-dataset-builder-1.0.0.jar`  
**Run:**
```bash
java -jar labpbr-dataset-builder-1.0.0.jar [options]
```
**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--inputDir` | Resource pack directory | `./resourcepacks` |
| `--outputDir` | Output directory | `./dataset` |
| `--maxSamples` | Max samples | `20000000` |
| `--seed` | Random seed | System time |

---

#### Module 2: Training

**JAR:** `tiny-mlp-training-1.0.0.jar`  
**Run:**
```bash
java -jar tiny-mlp-training-1.0.0.jar [options]
```
**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--data` | Data file | `train_data.bin` |
| `--labels` | Label file | `train_labels.bin` |
| `--output` | Model output path | `height_model.ser` |
| `--batch-size` | Batch size | `64` |
| `--epochs` | Max epochs | `50` |
| `--patience` | Early stop patience | `8` |
| `--lr` | Initial learning rate | `0.01` |
| `--lr-decay` | Decay factor | `0.9` |
| `--lr-step` | Decay step | `10` |
| `--seed` | Random seed | `42` |
| `--total-samples` | Total samples | `2000000` |
| `--train-size` | Training samples | `1000000` |
| `--val-size` | Validation samples | `20000` |
| `--help` | Show help | — |

---

#### Module 3: Inference

**JAR:** `pbr-inference-1.0.0.jar`  
**Run:**
```bash
java -jar pbr-inference-1.0.0.jar --input <input_image> [options]
```
**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Input image path | Required |
| `--outputDir` | Output directory | `./output` |
| `--model` | Model path | `height_model.ser` |
| `--strength` | Normal strength | `6.0` |
| `--pixelate` | Enable pixelation | `false` |
| `--smoothness` | Base smoothness | `0.2` |
| `--metallic` | Base metallic | `0.0` |
| `--invert-height` | Invert height | `true` |
| `--invert-normal-y` | Invert normal Y | `false` |
| `--height-strength` | Height contrast | `1.2` |
| `--height-min` | Height min | `0.2` |
| `--height-max` | Height max | `1.0` |
| `--height-smooth` | Blur radius | `2` |
| `--norm-percentile` | Percentile cutoff | `2.0` |
| `--help` | Show help | — |

---

Made with ❤️ by ENA