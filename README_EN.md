# 🛰️ Land Cover Classification System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Vue.js](https://img.shields.io/badge/Vue.js-3.x-4fc08d?style=for-the-badge&logo=vue.js&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

**An Intelligent Land Cover Classification System for High-Resolution Remote Sensing Images based on Deep Learning**

[English](README_EN.md) | [中文](README.md)

[Features](#-features) · [Quick Start](#-quick-start) · [Architecture](#-system-architecture) · [Core Models](#-core-models)

</div>

---

## 📖 Introduction

This project is a B/S architecture web system built with **Python FastAPI** and **Vue.js 3**, designed to achieve intelligent land cover classification for high-resolution remote sensing images. The system integrates multiple deep learning models and includes an exploratory improvement of the core **UNetMamba** model.

### 🎯 Background

With the rapid development of remote sensing technology, high-resolution remote sensing images play a crucial role in urban planning, environmental monitoring, disaster assessment, and other fields. This project aims to build a user-friendly intelligent analysis platform that integrates advanced deep learning algorithms to achieve efficient and accurate automated classification of remote sensing images.

---

## ✨ Features

| Feature | Description |
|:---------|:-----|
| 🔐 **User Management** | Secure registration and login authentication |
| 📤 **Image Processing** | Support for single/batch upload of remote sensing images with adaptive preprocessing |
| 🤖 **Model Segmentation** | Choose from UNetMamba, DC-Swin, UNetFormer, and other models for semantic segmentation |
| 📊 **Result Analysis** | Overlay comparison of original/segmented images, transparency adjustment, statistical chart generation |
| 📜 **History** | Automatically save detailed records of segmentation tasks |
| 🤝 **AI Assistant** | Integrated Large Language Model to assist in generating analysis reports |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Vue.js 3)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ Login/Reg   │  │ Image Upload│  │   Result Visualization  │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                         REST API
                              │
┌─────────────────────────────────────────────────────────────────┐
│                       Backend (FastAPI)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │    Auth     │  │ Processing  │  │    Model Inference      │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                     Model Layer (PyTorch)                       │
│  ┌───────────────┐  ┌────────────┐  ┌──────────────────────┐   │
│  │  UNetMamba    │  │  DC-Swin   │  │    UNetFormer        │   │
│  │  (with CA)    │  │  Small     │  │    R18               │   │
│  └───────────────┘  └────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Core Models

The core algorithm of this project is **UNetMamba**, which combines the multi-scale feature fusion capability of U-Net with the linear computational efficiency of the Mamba state space model.

### 🔬 Innovations

Introduced the **Coordinate Attention (CA) mechanism** at the input end of the original UNetMamba model:

- **Motivation**: To enhance the model's sensitivity to spatial structure and positional information of input images at the very beginning of feature extraction.
- **Implementation**: Input images flow through a CA module for attention weighting before entering the ResT encoder backbone network.

### 📈 Experimental Results

| Dataset | Model | mIoU (Excluding Background) |
|:-------|:-----|:---------------:|
| LoveDA | UNetMamba_CA | **59.66%** |
| Vaihingen | UNetMamba_CA | Outperforms Baseline |
| Potsdam | UNetMamba_CA | Outperforms Baseline |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- CUDA 11.x (Recommended for GPU acceleration)

### Backend Startup

```bash
# Enter backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env file, fill in SECRET_KEY, DEEPSEEK_API_KEY, etc.

# Download model weights to backend/model_weights directory
# (Please contact the maintainer for model weights)

# Start service
uvicorn main:app --reload --port 8000
```

### Frontend Startup

```bash
# Enter frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run serve
```

Visit http://localhost:8081 to use the system.

---

## 📁 Project Structure

```
Land-Cover-Classification-System/
├── backend/                  # Backend code
│   ├── main.py              # FastAPI main app
│   ├── database.py          # Database config
│   ├── db_models.py         # Data models
│   ├── schemas.py           # Pydantic schemas
│   └── requirements.txt     # Python dependencies
├── frontend/                 # Frontend code
│   ├── src/
│   │   ├── views/           # Page components
│   │   ├── components/      # Reusable components
│   │   └── router/          # Router config
│   └── package.json         # Node dependencies
├── UNetMamba/               # Model training code
└── README.md
```

---

## 🙏 Acknowledgments

The core model implementation of this project is based on the official **UNetMamba** model with modifications and experiments.

- **Original Paper**: *UNetMamba: An Efficient UNet-Like Mamba for Semantic Segmentation of High-Resolution Remote Sensing Images*
- **Original Repository**: [EnzeZhu2001/UNetMamba](https://github.com/EnzeZhu2001/UNetMamba)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

The use of UNetMamba follows its original Apache License 2.0 protocol.
