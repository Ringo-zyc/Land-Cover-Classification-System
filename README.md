# 🛰️ 基于遥感卫星图像的地表覆盖分类系统

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Vue.js](https://img.shields.io/badge/Vue.js-3.x-4fc08d?style=for-the-badge&logo=vue.js&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

**一个基于深度学习的高分辨率遥感图像智能化地表覆盖分类系统**

[English](README_EN.md) | [中文](README.md) | [日本語](README_JP.md)

[功能特性](#-系统功能) · [快速开始](#-快速开始) · [技术架构](#-系统架构) · [模型说明](#-核心模型)

</div>

---

## 📖 项目简介

本项目是一个基于 **Python FastAPI** 和 **Vue.js 3** 构建的前后端分离 Web 系统，旨在实现高分辨率遥感图像的智能化地表覆盖分类。系统整合了多种深度学习模型，并对核心的 **UNetMamba** 模型进行了探索性改进。

### 🎯 项目背景

随着遥感技术的飞速发展，高分辨率遥感图像已在城市规划、环境监测、灾害评估等领域发挥着至关重要的作用。本项目旨在构建一个用户友好的智能化分析平台，集成先进的深度学习算法，实现遥感图像的高效、精准自动化分类。

---

## ✨ 系统功能

| 功能模块 | 描述 |
|:---------|:-----|
| 🔐 **用户管理** | 安全的注册与登录认证 |
| 📤 **图像处理** | 支持单张/批量遥感图像上传，自适应预处理 |
| 🤖 **模型分割** | 可选择 UNetMamba、DC-Swin、UNetFormer 等模型执行语义分割 |
| 📊 **结果分析** | 原图/分割图叠加对比、透明度调节、统计图表生成 |
| 📜 **历史记录** | 自动保存分割任务详细记录 |
| 🤝 **AI 助手** | 集成大语言模型辅助生成分析报告 |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        前端 (Vue.js 3)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  登录/注册  │  │   图像上传   │  │   分割结果可视化展示     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                         REST API
                              │
┌─────────────────────────────────────────────────────────────────┐
│                       后端 (FastAPI)                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  用户认证   │  │  图像处理   │  │      模型调度推理        │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                     模型层 (PyTorch)                            │
│  ┌───────────────┐  ┌────────────┐  ┌──────────────────────┐   │
│  │  UNetMamba    │  │  DC-Swin   │  │    UNetFormer        │   │
│  │  (with CA)    │  │  Small     │  │    R18               │   │
│  └───────────────┘  └────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 核心模型

本项目的核心算法是 **UNetMamba**，结合了 U-Net 的多尺度特征融合能力和 Mamba 状态空间模型的线性计算效率优势。

### 🔬 创新改进

在原始 UNetMamba 模型的**输入端引入了坐标注意力 (Coordinate Attention, CA) 机制**：

- **改进动机**：在特征提取的最开始提升模型对输入图像空间结构和位置信息的敏感度
- **实现方式**：输入图像在进入 ResT 编码器主干网络之前，首先流经 CA 模块进行注意力加权

### 📈 实验结果

| 数据集 | 模型 | mIoU (不计背景) |
|:-------|:-----|:---------------:|
| LoveDA | UNetMamba_CA | **59.66%** |
| Vaihingen | UNetMamba_CA | 优于基准 |
| Potsdam | UNetMamba_CA | 优于基准 |

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- Node.js 16+
- CUDA 11.x (推荐用于 GPU 加速)

### 后端启动

```bash
# 进入后端目录
cd backend

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入 SECRET_KEY, DEEPSEEK_API_KEY 等

# 下载模型权重文件到 backend/model_weights 目录
# (请联系项目维护者获取模型权重)

# 启动服务
uvicorn main:app --reload --port 8000
```

### 前端启动

```bash
# 进入前端目录
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run serve
```

访问 http://localhost:8081 即可使用系统。

---

## 📁 项目结构

```
Land-Cover-Classification-System/
├── backend/                  # 后端代码
│   ├── main.py              # FastAPI 主应用
│   ├── database.py          # 数据库配置
│   ├── db_models.py         # 数据模型
│   ├── schemas.py           # Pydantic 模式
│   └── requirements.txt     # Python 依赖
├── frontend/                 # 前端代码
│   ├── src/
│   │   ├── views/           # 页面组件
│   │   ├── components/      # 复用组件
│   │   └── router/          # 路由配置
│   └── package.json         # Node 依赖
├── UNetMamba/               # 模型训练代码
└── README.md
```

---

## 🙏 致谢

本项目的核心模型实现基于官方 **UNetMamba** 模型进行了修改和实验。

- **原始论文**: *UNetMamba: An Efficient UNet-Like Mamba for Semantic Segmentation of High-Resolution Remote Sensing Images*
- **原始代码仓库**: [EnzeZhu2001/UNetMamba](https://github.com/EnzeZhu2001/UNetMamba)

---

## 📄 License

本项目采用 [MIT License](LICENSE) 许可证。

对 UNetMamba 的使用遵循其原始的 Apache License 2.0 协议。