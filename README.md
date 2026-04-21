# lung-cnn

CT画像から切り出したボクセルデータを用いて、3次元畳み込みニューラルネットワーク（3D-CNN）で肺がんの**胸膜浸潤**を判別する研究用リポジトリです。

本研究では、従来の2D-CNNによる判別手法に対して、CT画像の腫瘍周辺を3次元的に切り出した **32×32×32** のボクセルデータを入力とし、3D-CNNによる胸膜浸潤判別を行いました。

また、学習データ作成のために、DICOM画像を表示し、クリック位置周辺をボクセルとして保存するツールも開発しています。

---

## Overview

肺がんの胸膜浸潤は、治療方針や術式の決定に関わる重要な所見です。  
一方で、CT画像のみから胸膜浸潤を判別することは容易ではありません。

従来研究では2次元畳み込みニューラルネットワーク（2D-CNN）を用いた判別が行われていましたが、本研究では **3D-CNN** を適用し、CT画像の空間的な情報を活用することで、より高精度な判別を試みました。

特に、肺全体の画像をそのまま入力とするのではなく、**腫瘍と胸膜が接している周辺領域** に着目し、その周辺を切り出した局所的な3次元ボクセルデータを入力としています。

---

## Features

- DICOM画像の表示
- スライダーによるCTスライス切り替え
- クリック位置を中心とした32×32×32ボクセルデータ生成
- VoxNetベースの3D-CNNによる二値分類
- 感度・特異度を用いた評価
- 平滑化・エッジ処理などの前処理実験
- Dockerによる実行環境構築

---

## Repository Structure

```bash
.
├── annotation/
│   ├── dicom_viewer.py
│   └── clipping.py
│
├── models/
│   ├── voxnet.py
│   └── chatGPT.py
│
├── archive/
│   ├── change_voxel_size.py
│   ├── check_voxel.py
│   ├── dicom2voxel_converter.py
│   ├── jpeg2voxel_converter.py
│   ├── patient_distribution.py
│   └── voxel_padding.py
│
├── dicom/
│   ├── people1/
│   └── people2/
│
├── main.py
├── various_reprocessing.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── distribution.png
```

### `annotation/`

DICOM画像の表示、スライス切り替え、クリック座標取得、ボクセル切り出しに関するコードを配置しています。

### `models/`

3D-CNNモデル定義を配置しています。VoxNetベースの分類モデルを利用します。

### `archive/`

試作・未採用・検証用コードを保管するためのディレクトリです。

### `dicom/`

サンプルDICOMデータを配置しています。

### `main.py`

学習・評価を実行するメインスクリプトです。

### `various_reprocessing.py`

平滑化やエッジ処理などの前処理実験を行うためのスクリプトです。

---

## Method

### 1. Voxel Data Creation

本研究では、CT画像から腫瘍周辺の **32×32×32** ボクセルデータを作成します。

`annotation/dicom_viewer.py` を用いてDICOMフォルダを読み込むと、CT画像を表示できます。  
スライダーでスライスを切り替え、対象位置をクリックすると、`annotation/clipping.py` により周辺領域を切り出して **NumPy形式 (`.npy`)** のボクセルデータとして保存します。

### 2. Classification

保存したボクセルデータを用いて、PyTorchによる二値分類を行います。

- Positive class: 胸膜浸潤あり
- Negative class: 胸膜浸潤なし

分類モデルには、VoxNetを参考にした3D-CNNを使用しています。  
本研究では、異なるネットワーク深さを持つ2種類のモデルを比較しました。

### 3. Evaluation

評価指標には、医療系研究でよく用いられる以下を使用しています。

- Sensitivity（感度）
- Specificity（特異度）

加えて、実装上は以下の指標も扱えるようにしています。

- Accuracy
- Precision
- Recall
- F1 Score

---

## Training Settings

主な学習設定は以下の通りです。

- Framework: PyTorch
- Task: Binary classification
- Optimizer: Adam
- Loss function: CrossEntropyLoss
- Batch size: 50
- Epochs: 100〜300

※ 実際の学習条件は、スクリプト内の設定に応じて変更される場合があります。

---

## Dataset

本研究では、胸部CT画像から作成したボクセルデータを使用します。

- 胸膜浸潤あり: 200件
- 胸膜浸潤なし: 307件
- 学習データ : テストデータ = 8 : 2

各データは `.npy` 形式で保存し、学習・評価に利用します。

---

## Results

従来の2D-CNN手法に対して、3D-CNNを用いた手法は感度・特異度の両面で改善が見られました。

| Method | Sensitivity | Specificity |
| --- | ---: | ---: |
| Previous method | 0.6670 | 0.2400 |
| Model 1 | 0.7368 | 0.7031 |
| Model 2 | 0.7110 | 0.5263 |

また、前処理実験では、平滑化やエッジ処理を加えても全体として性能向上にはつながらず、**前処理なしの浅いモデル** が最も良好な結果となりました。

### Preprocessing Results

| Input | Sensitivity | Specificity |
| --- | ---: | ---: |
| No preprocessing | 0.7368 | 0.7031 |
| Pattern 1: Smoothing only | 0.5172 | 0.6712 |
| Pattern 2: Edge only | 0.4889 | 0.5179 |
| Pattern 3: Smoothing → Edge | 0.6750 | 0.4754 |
| Pattern 4: Edge → Smoothing | 0.2667 | 0.8194 |

---

## Discussion

本研究では、肺全体を入力とするのではなく、**腫瘍周辺に絞った局所3次元データ** を用いたことが精度向上に寄与したと考えられます。

また、より深いネットワーク構成が常に有効とは限らず、今回のタスクでは**比較的浅いモデル** の方が高い性能を示しました。

一方で、32×32×32と比較的小さいボクセルデータに対して平滑化やエッジ処理を施すと、重要な特徴が失われ、性能低下につながる可能性が示されました。

---

## Environment

### Local

```bash
pip install -r requirements.txt
python main.py
```

### Docker

```bash
docker compose up --build
```

---

## Usage

### 1. DICOM Viewer / Voxel Creation

```bash
python annotation/dicom_viewer.py
```

手順:

1. DICOMフォルダを選択
2. スライダーでスライスを移動
3. 切り出したい位置をクリック
4. 32×32×32のボクセルデータを `.npy` として保存

### 2. Training

保存したボクセルデータを学習用ディレクトリに配置した後、以下を実行します。

```bash
python main.py
```

### 3. Preprocessing Experiment

前処理実験を行う場合は、以下のスクリプトを利用します。

```bash
python various_reprocessing.py
```

---

## Notes

- 本リポジトリは研究・ポートフォリオ公開を目的としています。
- 医療画像データには個人情報・研究倫理上の配慮が必要なため、公開可能なサンプルのみ含めています。
- 再現実験には、元データセットおよび適切な研究環境が必要です。
- 一部のコードには試作段階の内容や実験用ファイルが含まれる場合があります。

---

## Future Work

- ボクセルデータサイズの見直し
- Grad-CAM等による判別根拠の可視化
- データ数の拡充
- モデル構成や前処理方法の改善
- 実用化に向けた精度向上

---

## Thesis

## Thesis
本研究の詳細は以下の卒業論文を参照してください。

- [卒業論文PDF](./docs/thesis.pdf)
