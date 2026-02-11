# 音楽ジャンル分類 - EfficientNet B7

音楽音声ファイルを10種類のジャンルに分類する深層学習モデルです。メルスペクトログラムを画像として扱い、事前学習済みEfficientNet B7でFine-tuningを行います。

## 📋 プロジェクト概要

- **タスク**: 音楽ジャンル分類（10クラス）
- **モデル**: EfficientNet B7 (ImageNet事前学習済み)
- **フレームワーク**: PyTorch
- **入力**: メルスペクトログラム（128×time_frames → 600×600にリサイズ）
- **出力**: 10クラス分類（blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock）

## 🎯 特徴

- **メルスペクトログラムベース**: 音声をメルスペクトログラムに変換し、画像として扱う
- **強力なデータオーギュメンテーション**:
  - Pitch Shifting（ピッチシフト）
  - Time Stretching（タイムストレッチ）
  - Adding Noise（ノイズ追加）
  - Time Shifting（タイムシフト）
  - SpecAugment（Time/Frequency Masking）
- **チェックポイント機能**: 学習の中断・再開に対応
- **メモリ効率**: メモリリーク対策を実装

## 📁 ディレクトリ構造

```
music_labeling/
├── train.py              # メイン学習スクリプト
├── eda.py                # 探索的データ分析
├── requirements.txt      # 依存パッケージ
├── README.md            # このファイル
├── data/
│   ├── train_master.csv      # 学習データのメタデータ
│   ├── label_master.csv      # ラベル定義
│   ├── sample_submit.csv     # 提出フォーマット例
│   ├── train_sound_*/        # 学習音声ファイル
│   └── test_sound_*/         # テスト音声ファイル
├── checkpoint.pth       # 学習チェックポイント（自動生成）
├── best_model.pth       # ベストモデル（自動生成）
└── submit.csv           # 提出ファイル（自動生成）
```

## 🚀 セットアップ

### 1. 仮想環境の作成と有効化

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2. 依存パッケージのインストール

```bash
# PyTorch (CUDA版) をインストール
# CUDA 12.1 の場合:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 の場合:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU版の場合:
pip install torch torchvision

# その他の依存パッケージ
pip install -r requirements.txt
```

### 3. GPU確認（オプション）

```bash
nvidia-smi  # GPU情報を確認
```

## 💻 使用方法

### 学習の実行

```bash
python train.py
```

### 学習の再開

学習が中断された場合、次回実行時に自動でチェックポイントから再開されます。

```bash
python train.py  # 自動でcheckpoint.pthから再開
```

### 最初からやり直す場合

```bash
rm checkpoint.pth best_model.pth
python train.py
```

## ⚙️ 設定パラメータ

`train.py` の設定セクションで以下を調整できます：

| パラメータ | デフォルト値 | 説明 |
|-----------|------------|------|
| `N_MELS` | 128 | メルバンド数 |
| `SR` | 22050 | サンプリングレート |
| `IMG_SIZE` | 600 | リサイズ後の画像サイズ |
| `BATCH_SIZE` | 24 | バッチサイズ（RTX3090向け） |
| `EPOCHS` | 50 | 最大エポック数 |
| `LR` | 2e-4 | 学習率 |
| `WEIGHT_DECAY` | 1e-4 | 重み減衰 |
| `NUM_CLASSES` | 10 | 分類クラス数 |
| `TEST_SIZE` | 0.1 | 検証データの割合 |

### バッチサイズの調整

GPUメモリに応じて調整してください：

- **RTX3090 (24GB)**: `BATCH_SIZE = 16-24`
- **RTX3080 (10GB)**: `BATCH_SIZE = 8-12`
- **RTX3060 (12GB)**: `BATCH_SIZE = 8-12`
- **CPU**: `BATCH_SIZE = 2-4`

## 📊 データオーギュメンテーション

以下のオーギュメンテーションが実装されています：

1. **Pitch Shifting**: ±2半音の範囲でランダムにピッチを変更
2. **Time Stretching**: 0.8～1.2倍の範囲で速度を変更（ピッチ維持）
3. **Adding Noise**: ガウシアンノイズを追加
4. **Time Shifting**: 最大0.2秒前後にシフト
5. **SpecAugment**: Time/Frequency Masking

各オーギュメンテーションは50%の確率で適用されます。

## 📈 学習の流れ

1. **データ読み込み**: 音声ファイルからメルスペクトログラムを抽出
2. **前処理**: パディング、標準化、リサイズ（600×600）
3. **Train/Valid分割**: 10%を検証データとして分割
4. **学習**: EfficientNet B7でFine-tuning
5. **評価**: 検証データで精度を評価
6. **推論**: テストデータで予測
7. **提出ファイル生成**: `submit.csv`を出力

## 📝 出力ファイル

- `checkpoint.pth`: 学習チェックポイント（ベストモデル更新時と中断時に保存）
- `best_model.pth`: ベストモデル（検証精度が最高のモデル）
- `submit.csv`: テストデータの予測結果

## 🔧 トラブルシューティング

### メモリ不足エラー

- `BATCH_SIZE`を小さくする
- `IMG_SIZE`を小さくする（例: 600 → 224）

### CUDA out of memory

```python
# train.py の設定を変更
BATCH_SIZE = 8  # より小さな値に
```

### 学習が遅い

- GPUが使用されているか確認: `nvidia-smi`
- `num_workers`を増やす（ただしWindowsでは0推奨）

## 📚 参考資料

- [librosa Documentation](https://librosa.org/doc/latest/index.html)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## 📄 ライセンス

このプロジェクトは学習・研究目的で使用してください。

## 👤 作成者

SIGNATE 音楽ジャンル分類コンペティション用実装
