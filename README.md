# MCPG - Minecraft PBR Generator

![License](https://img.shields.io/badge/License-GPLv3-blue.svg)
![Java](https://img.shields.io/badge/Java-11%2B-red.svg)
![Build](https://img.shields.io/badge/Build-Maven-C71A36.svg)
![Release](https://img.shields.io/badge/Release-v1.1.0-brightgreen)

---

## English

### Description

This project is a proof-of-concept tool that uses a small neural network to automatically generate LabPBR maps from Minecraft textures.

The neural network predicts only the height map from the texture. The normal map is derived from the height map via a Sobel‑style difference algorithm, while roughness and metalness are mapped directly from the grayscale and red channel of the original image.

The project is divided into three independent modules responsible for data preparation, model training, and inference generation, which can be used separately as needed.

---

### Quick Start

```bash
# Build dataset
java -jar labpbr-dataset-builder-1.1.0.jar

# Train model
java -jar tiny-mlp-training-1.1.0.jar

# Generate PBR textures
java -jar pbr-inference-1.1.0.jar --input ./texture.png
```

All modules automatically load `config.yml` from the working directory. Command-line arguments override values in the config file.

---

### Configuration File

`config.yml` contains all runtime parameters. Below is the complete list.

#### `global`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | long | `11451` | Random seed for sampling and augmentation |
| `feature_dim` | int | `100` | Feature vector dimension |
| `label_dim` | int | `5` | Label vector dimension |
| `patch_size` | int | `5` | Patch size for feature extraction |

#### `dataset_builder`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dir` | string | `./resourcepacks` | Directory containing resource packs |
| `output_dir` | string | `./dataset` | Output directory for binary dataset |
| `max_samples` | int | `120000` | Maximum samples to extract |
| `target_texture_size` | int | `128` | Resize textures to this size |
| `normal_overflow_ratio` | float | `0.5` | Maximum allowed overflow ratio for normal maps |

#### `training`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_path` | string | `./dataset/train_data.bin` | Path to feature data file |
| `label_path` | string | `./dataset/train_labels.bin` | Path to label data file |
| `model_output` | string | `./height_model.ser` | Path to save trained model |
| `total_samples` | int | `120000` | Total samples in dataset |
| `train_split` | int | `100000` | Number of training samples |
| `val_split` | int | `20000` | Number of validation samples |

`train_split + val_split` must be less than or equal to `total_samples`.

##### `hyperparams`

| Parameter | Type | Default | Description | Constraint |
|-----------|------|---------|-------------|------------|
| `batch_size` | int | `128` | Batch size for training | |
| `epochs` | int | `100` | Maximum training epochs | |
| `patience` | int | `8` | Early stopping patience | |
| `learning_rate` | float | `0.005` | Initial learning rate | |
| `lr_decay` | float | `0.95` | Learning rate decay factor | |
| `lr_step` | int | `10` | Epochs per learning rate decay | |
| `layers` | int array | `[100, 32, 16, 1]` | Network architecture | First element must equal `feature_dim`; last must be `1` |

#### `inference`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | string | `./height_model.ser` | Path to trained model |
| `output_dir` | string | `./output` | Output directory |
| `normal_strength` | float | `8.0` | Normal map intensity |
| `pixelate` | boolean | `false` | Enable hard-edge pixelation |
| `base_smoothness` | float | `0.3` | Base smoothness value |
| `base_metallic` | float | `0.1` | Base metallic value |

##### `height`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `invert` | boolean | `true` | Invert height map |
| `strength` | float | `1.5` | Height contrast multiplier |
| `min` | float | `0.1` | Minimum height value |
| `max` | float | `1.0` | Maximum height value |
| `smooth_radius` | int | `4` | Box blur radius for height map |
| `norm_percentile` | float | `1.5` | Percentile cutoff for normalization |

##### `normal`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `invert_y` | boolean | `true` | Invert Y component of normal map |

---

### Module Details

#### Module 1: Dataset Builder

**Input:** Directory containing Minecraft resource packs (folders or `.zip` files). Each texture group must contain three files with the same base name: `{name}.png`, `{name}_n.png`, and `{name}_s.png`.

**Output:** Two binary files in the output directory:
- `train_data.bin` – Feature vectors
- `train_labels.bin` – Label vectors

**Exit codes:** `0` on success, `1` on failure.

#### Module 2: Training

**Input:** Binary dataset files generated by Module 1.

**Output:** Serialized model file (`height_model.ser` by default).

**Exit codes:** `0` on success, `1` on failure.

#### Module 3: Inference

**Input:** PNG image file (RGB or ARGB). The image is automatically resized to at least 128×128 if smaller.

**Output:** Two PNG files in the output directory:
- `texture_n.png` – Normal map (RGBA)
- `texture_s.png` – Smoothness and metallic map (RGBA)

The `--input` parameter is required.

**Exit codes:** `0` on success, `1` on failure.

---

### Command-line Overrides

Any parameter can be overridden via command line. Examples:

```bash
# Override normal strength
java -jar pbr-inference-1.1.0.jar --input ./texture.png --strength 10.0

# Override max samples
java -jar labpbr-dataset-builder-1.1.0.jar --maxSamples 50000

# Override learning rate
java -jar tiny-mlp-training-1.1.0.jar --lr 0.001
```

Use `--help` to see all available options for each module.

---

Made with ❤️ by ENA

---

---

## 中文

### 描述

本项目是一个概念验证工具，利用小型神经网络从 Minecraft 纹理自动生成 LabPBR 贴图。

神经网络只从纹理预测高度图。法线贴图通过 Sobel 风格的差分算法从高度图推导，粗糙度和金属度直接从原图的灰度和红色通道映射生成。

项目分为三个独立模块，分别负责数据准备、模型训练和推理生成，可以根据需要单独使用。

---

### 快速上手

```bash
# 构建数据集
java -jar labpbr-dataset-builder-1.1.0.jar

# 训练模型
java -jar tiny-mlp-training-1.1.0.jar

# 生成 PBR 贴图
java -jar pbr-inference-1.1.0.jar --input ./texture.png
```

所有模块会自动加载工作目录下的 `config.yml`。命令行参数会覆盖配置文件中的值。

---

### 配置文件

`config.yml` 包含所有运行参数。完整列表如下。

#### `global`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `seed` | long | `11451` | 采样和数据增强的随机种子 |
| `feature_dim` | int | `100` | 特征向量维度 |
| `label_dim` | int | `5` | 标签向量维度 |
| `patch_size` | int | `5` | 特征提取的邻域大小 |

#### `dataset_builder`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_dir` | string | `./resourcepacks` | 资源包存放目录 |
| `output_dir` | string | `./dataset` | 二进制数据集输出目录 |
| `max_samples` | int | `120000` | 最大采样数量 |
| `target_texture_size` | int | `128` | 纹理缩放目标尺寸 |
| `normal_overflow_ratio` | float | `0.5` | 法线贴图最大溢出比例 |

#### `training`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data_path` | string | `./dataset/train_data.bin` | 特征数据文件路径 |
| `label_path` | string | `./dataset/train_labels.bin` | 标签数据文件路径 |
| `model_output` | string | `./height_model.ser` | 模型保存路径 |
| `total_samples` | int | `120000` | 数据集中总样本数 |
| `train_split` | int | `100000` | 训练集样本数 |
| `val_split` | int | `20000` | 验证集样本数 |

`train_split + val_split` 必须小于或等于 `total_samples`。

##### `hyperparams`

| 参数 | 类型 | 默认值 | 说明 | 约束 |
|------|------|--------|------|------|
| `batch_size` | int | `128` | 训练批次大小 | |
| `epochs` | int | `100` | 最大训练轮数 | |
| `patience` | int | `8` | 早停耐心值 | |
| `learning_rate` | float | `0.005` | 初始学习率 | |
| `lr_decay` | float | `0.95` | 学习率衰减系数 | |
| `lr_step` | int | `10` | 学习率衰减步长 | |
| `layers` | int 数组 | `[100, 32, 16, 1]` | 网络结构 | 第一个元素必须等于 `feature_dim`，最后一个必须为 `1` |

#### `inference`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_path` | string | `./height_model.ser` | 训练好的模型路径 |
| `output_dir` | string | `./output` | 输出目录 |
| `normal_strength` | float | `8.0` | 法线贴图强度 |
| `pixelate` | boolean | `false` | 启用硬边缘像素化 |
| `base_smoothness` | float | `0.3` | 基础光滑度 |
| `base_metallic` | float | `0.1` | 基础金属度 |

##### `height`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `invert` | boolean | `true` | 反转高度图 |
| `strength` | float | `1.5` | 高度对比度增强倍数 |
| `min` | float | `0.1` | 高度最小值 |
| `max` | float | `1.0` | 高度最大值 |
| `smooth_radius` | int | `4` | 高度图均值模糊半径 |
| `norm_percentile` | float | `1.5` | 归一化百分位截断 |

##### `normal`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `invert_y` | boolean | `true` | 翻转法线 Y 轴分量 |

---

### 模块详解

#### 模块一：数据集构建

**输入：** 存放 Minecraft 资源包的目录（文件夹或 `.zip` 文件）。每个纹理组必须包含三个同名基础文件：`{name}.png`、`{name}_n.png`、`{name}_s.png`。

**输出：** 输出目录中的两个二进制文件：
- `train_data.bin` – 特征向量
- `train_labels.bin` – 标签向量

**退出码：** `0` 表示成功，`1` 表示失败。

#### 模块二：训练

**输入：** 模块一生成的二进制数据集文件。

**输出：** 序列化模型文件（默认 `height_model.ser`）。

**退出码：** `0` 表示成功，`1` 表示失败。

#### 模块三：推理

**输入：** PNG 图片文件（RGB 或 ARGB）。如果图片小于 128×128，会自动放大。

**输出：** 输出目录中的两个 PNG 文件：
- `texture_n.png` – 法线贴图
- `texture_s.png` – 光滑度和金属度贴图

`--input` 参数为必需。

**退出码：** `0` 表示成功，`1` 表示失败。

---

### 命令行参数覆盖

任何参数都可以通过命令行覆盖。示例：

```bash
# 覆盖法线强度
java -jar pbr-inference-1.1.0.jar --input ./texture.png --strength 10.0

# 覆盖最大样本数
java -jar labpbr-dataset-builder-1.1.0.jar --maxSamples 50000

# 覆盖学习率
java -jar tiny-mlp-training-1.1.0.jar --lr 0.001
```

使用 `--help` 可查看各模块的所有可用选项。

---

Made with ❤️ by ENA