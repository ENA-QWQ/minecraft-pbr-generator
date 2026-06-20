# MCPG - Minecraft PBR Generator

[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE) [![Java](https://img.shields.io/badge/Java-8%2B-red.svg)](https://adoptium.net/) [![Maven](https://img.shields.io/badge/Build-Maven-C71A36.svg)](https://maven.apache.org/)

本项目是一个概念验证工具，利用小型神经网络从 MC 纹理自动生成 LabPBR 贴图。<br>

神经网络只从纹理预测高度图。法线贴图通过 Sobel 风格的差分算法从高度图推导，粗糙度和金属度直接从原图的灰度和红色通道映射生成。（以后会改进高光贴图的生成算法）<br>

项目分为三个独立模块，分别负责数据准备、模型训练和推理生成，可以根据需要单独使用。<br>

不追求生产级精度或性能。模型结构轻量，仅在有限数据集上测试，输出效果可能存在瑕疵。欢迎用于学习、实验或二次开发。

---

This project is a proof-of-concept tool that uses a small neural network to automatically generate LabPBR maps from MC textures.<br>

The neural network predicts only the height map from the texture. The normal map is derived from the height map via a Sobel‑style difference algorithm, while roughness and metalness are mapped directly from the grayscale and red channel of the original image. (The specular map generation will be improved in the future.)<br>

The project is divided into three independent modules responsible for data preparation, model training, and inference generation, which can be used separately as needed.<br>

It does not aim for production‑grade accuracy or performance. The model architecture is lightweight and has only been tested on a limited dataset, so the output may have imperfections. It is welcome for learning, experimentation, or further development.

---

Made with ❤️ by ENA