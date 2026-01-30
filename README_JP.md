# 🛰️ 衛星画像地表面被覆分類システム (Land Cover Classification System)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Vue.js](https://img.shields.io/badge/Vue.js-3.x-4fc08d?style=for-the-badge&logo=vue.js&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

**ディープラーニングに基づく高解像度リモートセンシング画像のためのインテリジェント地表面被覆分類システム**

[English](README_EN.md) | [中文](README.md) | [日本語](README_JP.md)

[機能一覧](#-機能一覧) · [クイックスタート](#-クイックスタート) · [アーキテクチャ](#-システムアーキテクチャ) · [モデル](#-コアモデル)

</div>

---

## 📖 はじめに

本プロジェクトは、**Python FastAPI** と **Vue.js 3** を用いて構築された前後分離型のWebシステムであり、高解像度リモートセンシング画像のインテリジェントな地表面被覆分類を実現することを目的としています。このシステムは複数のディープラーニングモデルを統合し、コアとなる **UNetMamba** モデルに対して探索的な改良を行っています。

### 🎯 背景

リモートセンシング技術の急速な発展に伴い、高解像度リモートセンシング画像は都市計画、環境モニタリング、災害評価などの分野で極めて重要な役割を果たしています。本プロジェクトは、高度なディープラーニングアルゴリズムを統合した使いやすい分析プラットフォームを構築し、リモートセンシング画像の効率的かつ高精度な自動分類を実現することを目指しています。

---

## ✨ 機能一覧

| 機能モジュール | 説明 |
|:---------|:-----|
| 🔐 **ユーザー管理** | 安全なユーザー登録とログイン認証 |
| 📤 **画像処理** | リモートセンシング画像のシングル/バッチアップロード、適応型前処理に対応 |
| 🤖 **モデル分割** | UNetMamba、DC-Swin、UNetFormer などのモデルを選択してセマンティックセグメンテーションを実行可能 |
| 📊 **結果分析** | 元画像/分割画像のオーバーレイ比較、透明度調整、統計グラフ生成 |
| 📜 **履歴記録** | 分割タスクの詳細記録を自動保存 |
| 🤝 **AI アシスタント** | 大規模言語モデル（LLM）を統合し、分析レポートの作成を支援 |

---

## 🏗️ システムアーキテクチャ

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Vue.js 3)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  ログイン   │  │ 画像アップ  │  │   結果可視化表示        │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                         REST API
                              │
┌─────────────────────────────────────────────────────────────────┐
│                       Backend (FastAPI)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │    認証     │  │   画像処理  │  │    モデル推論           │ │
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

## 🧠 コアモデル

本プロジェクトのコアアルゴリズムは **UNetMamba** です。これは U-Net のマルチスケール特徴融合能力と、Mamba 状態空間モデルの線形計算効率の利点を組み合わせています。

### 🔬 技術的改良

オリジナルの UNetMamba モデルの**入力端に座標注意機構 (Coordinate Attention, CA) を導入**しました。

- **改良の動機**: 特徴抽出の初期段階において、入力画像の空間構造と位置情報に対するモデルの感度を向上させるため。
- **実装方法**: 入力画像は ResT エンコーダーのバックボーンネットワークに入る前に、まず CA モジュールを通過し、注意重み付けが行われます。

### 📈 実験結果

| データセット | モデル | mIoU (背景を除く) |
|:-------|:-----|:---------------:|
| LoveDA | UNetMamba_CA | **59.66%** |
| Vaihingen | UNetMamba_CA | ベースラインを上回る |
| Potsdam | UNetMamba_CA | ベースラインを上回る |

---

## 🚀 クイックスタート

### 必要環境

- Python 3.8+
- Node.js 16+
- CUDA 11.x (GPU高速化に推奨)

### バックエンド起動

```bash
# バックエンドディレクトリに移動
cd backend

# 仮想環境の作成
python -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate

# 依存関係のインストール
pip install -r requirements.txt

# 環境変数の設定
cp .env.example .env
# .env ファイルを編集し、SECRET_KEY, DEEPSEEK_API_KEY などを入力

# モデルの重みファイルを backend/model_weights ディレクトリにダウンロード
# (モデルの重みについては管理者に問い合わせてください)

# サービスの起動
uvicorn main:app --reload --port 8000
```

### フロントエンド起動

```bash
# フロントエンドディレクトリに移動
cd frontend

# 依存関係のインストール
npm install

# 開発サーバーの起動
npm run serve
```

http://localhost:8081 にアクセスしてシステムを使用できます。

---

## 📁 プロジェクト構造

```
Land-Cover-Classification-System/
├── backend/                  # バックエンドコード
│   ├── main.py              # FastAPI メインアプリ
│   ├── database.py          # データベース設定
│   ├── db_models.py         # データモデル
│   ├── schemas.py           # Pydanticスキーマ
│   └── requirements.txt     # Python依存関係
├── frontend/                 # フロントエンドコード
│   ├── src/
│   │   ├── views/           # ページコンポーネント
│   │   ├── components/      # 再利用可能コンポーネント
│   │   └── router/          # ルーティング設定
│   └── package.json         # Node依存関係
├── UNetMamba/               # モデル学習コード
└── README.md
```

---

## 🙏 謝辞

本プロジェクトのコアモデル実装は、公式の **UNetMamba** モデルに基づいて修正および実験を行ったものです。

- **原論文**: *UNetMamba: An Efficient UNet-Like Mamba for Semantic Segmentation of High-Resolution Remote Sensing Images*
- **リポジトリ**: [EnzeZhu2001/UNetMamba](https://github.com/EnzeZhu2001/UNetMamba)

---

## 📄 ライセンス

本プロジェクトは [MIT License](LICENSE) の下でライセンスされています。

UNetMamba の使用については、オリジナルの Apache License 2.0 プロトコルに従います。
