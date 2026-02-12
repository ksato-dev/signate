# 音楽ジャンル分類 - EfficientNet B3

音楽音声ファイルを10種類のジャンルに分類する深層学習モデルです。
Demucs で音源を4ステム分離し、4チャンネルメルスペクトログラムを EfficientNet B3 で Fine-tuning します。

## プロジェクト概要

- **タスク**: 音楽ジャンル分類（10クラス）
- **モデル**: EfficientNet B3 (ImageNet事前学習済み)
- **音源分離**: Demucs (htdemucs) — 4ステム分離
- **フレームワーク**: PyTorch
- **入力**: 4チャンネルメルスペクトログラム（Vocals / Drums / Bass / Other → 600x600にリサイズ）
- **出力**: 10クラス分類（blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock）

## モデルアーキテクチャ

```
Input: (B, 4, H, W)   ← 4ch メルスペクトログラム（vocals / drums / bass / other）
        ↓
EfficientNet B3（features[:6] 凍結）
        ↓
Classifier:
  Dropout(0.5) → Linear(1536, 512) → ReLU → Dropout(0.3) → Linear(512, 10)
        ↓
Output logits
```

### 4チャンネル入力の初期化

EfficientNet B3 は ImageNet で3チャンネル（RGB）入力として事前学習されています。
4チャンネル入力に対応するため、最初の Conv 層を以下のように拡張しています。

```
新Conv.weight[:, :3] = 旧Conv.weight     ← ImageNet 事前学習済みの重みをそのまま使用
新Conv.weight[:, 3]  = 旧Conv.weight の3ch平均  ← 4ch 目は既存3ch の平均で初期化
```

### 設計ポイント

| 要素 | 内容 |
|---|---|
| EfficientNet B3 | ImageNet 事前学習済みの CNN。`features[:6]` を凍結して過学習を防止 |
| 4stem 入力 | Vocals / Drums / Bass / Other を個別チャンネルとして入力。楽器ごとの特徴を保持 |
| カスタム分類ヘッド | Dropout + Linear(1536→512) + ReLU + Dropout + Linear(512→10) |
| Mixed Precision (AMP) | float16 で計算し VRAM 使用量を約40-50%削減 |

## 特徴

- **4ステム音源分離 (Demucs)**: Vocals / Drums / Bass / Other に分離し、4チャンネル入力として楽器ごとの情報を保持
- **Mixed Precision Training (AMP)**: VRAM 使用量を大幅に削減
- **強力なデータオーギュメンテーション**:
  - Pitch Shifting（ピッチシフト）
  - Time Stretching（タイムストレッチ）
  - Adding Noise（ノイズ追加）
  - Time Shifting（タイムシフト）
  - SpecAugment（Time/Frequency Masking）
- **チェックポイント機能**: 学習の中断・再開に対応
- **ログ出力**: コンソールとファイル（`train_log.txt`）への同時出力
- **メモリ効率**: Demucs 分離結果のキャッシュ、GPU メモリリーク対策

## 処理フロー

1. **音源分離**: Demucs で4ステム（vocals / drums / bass / other）に分離（キャッシュ付き）
2. **メルスペクトログラム抽出**: 4チャンネル (128 x time_frames) を生成
3. **前処理**: パディング、チャンネルごと標準化、リサイズ (600x600)
4. **Train/Valid 分割**: 10%を検証データとして分割
5. **学習**: features[:6] 凍結で分類ヘッドを学習（CosineAnnealing + Early Stopping）
6. **推論**: テストデータで予測
7. **提出ファイル生成**: `submit.csv` を出力

## ディレクトリ構造

```
music_labeling/
├── train.py                  # メイン学習スクリプト
├── config.py                 # 設定ファイル（ハイパーパラメータ管理）
├── export_separated_wav.py   # 音源分離キャッシュ → WAV 変換ツール
├── eda.py                    # 探索的データ分析
├── requirements.txt          # 依存パッケージ
├── README.md                 # このファイル
├── data/
│   ├── train_master.csv      # 学習データのメタデータ
│   ├── label_master.csv      # ラベル定義
│   ├── sample_submit.csv     # 提出フォーマット例
│   ├── separated/            # 音源分離キャッシュ（自動生成, 4stem .npz）
│   ├── train_sound_*/        # 学習音声ファイル
│   └── test_sound_*/         # テスト音声ファイル
├── checkpoint.pth            # 学習チェックポイント（自動生成）
├── best_model.pth            # ベストモデル（自動生成）
├── train_log.txt             # 学習ログ（自動生成）
└── submit.csv                # 提出ファイル（自動生成）
```

## セットアップ

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

## 使用方法

### 学習の実行

```bash
cd music_labeling
python train.py
```

学習ログは `train_log.txt` にも自動的に出力されます。

### 学習の再開

学習が中断された場合、`config.py` で `RESUME_FROM_CHECKPOINT = True` に設定して再実行します。

```python
RESUME_FROM_CHECKPOINT = True      # チェックポイントから再開するか
LOAD_OPTIMIZER_STATE = False       # 新実装ではFalse推奨
LOAD_SCHEDULER_STATE = False       # 新実装ではFalse推奨
```

### 最初からやり直す場合

```bash
rm checkpoint.pth best_model.pth
python train.py
```

または `config.py` で `RESUME_FROM_CHECKPOINT = False` に設定。

### 音源分離結果の WAV 出力

```bash
# 全ファイルを変換
python export_separated_wav.py

# 特定のファイルだけ変換
python export_separated_wav.py train_0 train_1

# パスを直接指定
python export_separated_wav.py data/separated/train_1.npz

# 特定のチャンネルだけ出力
python export_separated_wav.py --channels vocals drums
```

## 設定パラメータ（config.py）

すべてのハイパーパラメータは `config.py` で管理しています。

### 基本設定

| パラメータ | デフォルト値 | 説明 |
|---|---|---|
| `SEED` | 42 | 乱数シード |
| `N_MELS` | 128 | メルバンド数 |
| `SR` | 22050 | サンプリングレート |
| `IMG_SIZE` | 600 | リサイズ後の画像サイズ |
| `NUM_CLASSES` | 10 | 分類クラス数 |
| `NUM_CHANNELS` | 4 | 入力チャンネル数（4stem） |

### 学習設定

| パラメータ | デフォルト値 | 説明 |
|---|---|---|
| `BATCH_SIZE` | 24 | バッチサイズ |
| `EPOCHS` | 50 | 最大エポック数 |
| `LR` | 2e-4 | 学習率 |
| `WEIGHT_DECAY` | 1e-4 | 重み減衰 |
| `TEST_SIZE` | 0.1 | 検証データの割合 |
| `PATIENCE` | 10 | Early Stopping の patience |
| `USE_AMP` | True | Mixed Precision Training |

### 音源分離設定

| パラメータ | デフォルト値 | 説明 |
|---|---|---|
| `SEPARATION_MODEL` | htdemucs | Demucs モデル名 |
| `SEPARATION_CACHE_DIR` | data/separated | 分離結果のキャッシュ先 |

### データオーギュメンテーション設定

| パラメータ | デフォルト値 | 説明 |
|---|---|---|
| `AUGMENT_PITCH_SHIFT` | True | ピッチシフト（±2半音） |
| `AUGMENT_TIME_STRETCH` | True | タイムストレッチ（0.8〜1.2倍） |
| `AUGMENT_ADD_NOISE` | True | ガウシアンノイズ追加 |
| `AUGMENT_TIME_SHIFT` | True | タイムシフト（±0.2秒） |
| `AUGMENT_TIME_MASKING` | True | SpecAugment Time Masking |
| `AUGMENT_FREQUENCY_MASKING` | True | SpecAugment Frequency Masking |

各オーギュメンテーションは50%の確率で適用されます。4チャンネル（vocals / drums / bass / other）に同一パラメータを適用し、チャンネル間の整合性を維持します。

### バッチサイズの調整目安

GPUメモリに応じて `config.py` の `BATCH_SIZE` を調整してください：

- **RTX3090 (24GB)**: 16-24
- **RTX3080 (10GB)**: 8-16
- **RTX3060 (12GB)**: 8-16
- **CPU**: 2-4

## 出力ファイル

| ファイル | 説明 |
|---|---|
| `checkpoint.pth` | 学習チェックポイント（ベストモデル更新時・中断時に保存） |
| `best_model.pth` | ベストモデル（検証精度が最高のモデル） |
| `submit.csv` | テストデータの予測結果（提出用） |
| `train_log.txt` | 学習ログ（コンソール出力と同一内容） |
| `data/separated/*.npz` | 音源分離のキャッシュ（vocals / drums / bass / other の波形） |

## トラブルシューティング

### CUDA out of memory

`config.py` の `BATCH_SIZE` を小さくしてください。

```python
BATCH_SIZE = 8  # メモリ不足時に下げる
```

`IMG_SIZE` を小さくすることでも対応可能です（例: 600 → 300）。
`USE_AMP = True` が有効になっていることも確認してください。

### 学習が遅い

- GPUが使用されているか確認: `nvidia-smi`
- スクリプト実行時に `Device: cuda` と表示されることを確認
- `num_workers` は Windows では 0 推奨

### 音源分離に時間がかかる

初回実行時のみ Demucs による音源分離が実行されます。分離結果は `data/separated/` にキャッシュされるため、2回目以降はスキップされます。

旧3ch形式のキャッシュが残っている場合は自動的に検出され、4stem 形式で再分離されます。

### チェックポイントの互換性エラー

モデル構造を変更した場合、古いチェックポイントは読み込めません。

```bash
rm checkpoint.pth best_model.pth
```

で削除してから再学習してください。

## 参考資料

- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Demucs - Music Source Separation](https://github.com/facebookresearch/demucs)
- [librosa Documentation](https://librosa.org/doc/latest/index.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## ライセンス

このプロジェクトは学習・研究目的で使用してください。

## 作成者

SIGNATE 音楽ジャンル分類コンペティション用実装
