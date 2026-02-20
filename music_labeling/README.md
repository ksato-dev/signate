# 音楽ジャンル分類 - EfficientNet B3

音楽音声ファイルを10種類のジャンルに分類する深層学習モデルです。
Demucs で音源を分離し、3チャンネルメルスペクトログラム（原音 / 伴奏 / ボーカル）を EfficientNet B3 で Fine-tuning します。

## プロジェクト概要

- **タスク**: 音楽ジャンル分類（10クラス）
- **モデル**: EfficientNet B3 (ImageNet事前学習済み)
- **音源分離**: Demucs (htdemucs) — 4ステム分離 → 3チャンネルに変換
- **フレームワーク**: PyTorch
- **入力**: 3チャンネルメルスペクトログラム（Original / Accompaniment / Vocals → 600x600にリサイズ）
- **出力**: 10クラス分類（blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock）

## モデルアーキテクチャ

```
Input: (B, 3, H, W)   ← 3ch メルスペクトログラム（original / accompaniment / vocals）
        ↓
EfficientNet B3（features[:6] 凍結）
        ↓
Classifier:
  Dropout(0.5) → Linear(1536, 512) → ReLU → Dropout(0.3) → Linear(512, 10)
        ↓
Output logits
```

3チャンネル入力は EfficientNet B3 の標準 RGB 入力とそのまま一致するため、Conv 層の改変は不要です。

### 設計ポイント

| 要素 | 内容 |
|---|---|
| EfficientNet B3 | ImageNet 事前学習済みの CNN。`features[:6]` を凍結して過学習を防止 |
| 3ch 入力 | Original / Accompaniment / Vocals を個別チャンネルとして入力 |
| カスタム分類ヘッド | Dropout + Linear(1536→512) + ReLU + Dropout + Linear(512→10) |
| Mixed Precision (AMP) | float16 で計算し VRAM 使用量を約40-50%削減 |

## 処理フロー

```
1. 音源分離 (Demucs 4stem → 3ch: original / accompaniment / vocals)
        ↓
2. メルスペクトログラム抽出 (128 mel bands)
        ↓
3. 前処理 (パディング → チャンネルごと標準化 → 600x600 リサイズ)
        ↓
4. K-Fold 交差検証 (StratifiedKFold, N_FOLDS=5)
   ├── Fold ごとにモデル学習 + Early Stopping
   └── CV スコアサマリー表示
        ↓  ← Ctrl+C で中止可能 (20秒待機)
5. 全データで最終モデルを再学習 (CV 平均ベストエポック数)
        ↓
6. テスト推論 → submit.csv 出力
```

`N_FOLDS = 0` に設定すると CV をスキップし、全データで直接学習します。

## 外部データ: MagnaTagATune

MagnaTagATune データセットの音声ファイルを、ジャンルタグに基づいて学習データに追加できます。

- `annotations_final.csv` の 10 ジャンルカラムを `label_master.csv` と照合
- 複数ジャンルタグを持つクリップは最も件数が少ないジャンルに優先割り当て
- クラスごとに均等サンプリング（デフォルト 50 件/クラス = 500 件追加）
- `USE_MAGNA_DATA = False` で無効化可能
- サンプリング結果は `data/magna_sampled.csv` に出力

## 外部データ: GTZAN

GTZAN Dataset (`genres_original`) の音声ファイルを学習データに追加できます。

- 10ジャンル × 100ファイル = 1000件（完全均等バランス）
- フォルダ名が `label_master.csv` のジャンル名と完全一致するため全件そのまま使用
- `USE_GTZAN_DATA = False` で無効化可能
- プロファイルは `data/gtzan_profile.csv` に出力

## ディレクトリ構造

```
music_labeling/
├── train.py                  # メイン学習スクリプト
├── config.py                 # 設定ファイル（ハイパーパラメータ管理）
├── export_separated_wav.py   # 音源分離キャッシュ → WAV 変換ツール
├── README.md                 # このファイル
├── data/
│   ├── train_master.csv      # 学習データのメタデータ
│   ├── label_master.csv      # ラベル定義
│   ├── sample_submit.csv     # 提出フォーマット例
│   ├── separated/            # 音源分離キャッシュ（自動生成, 4stem .npz）
│   ├── precomputed_mel.npz   # メルスペクトログラムキャッシュ（自動生成）
│   ├── augmentation_profile.csv  # オーギュメンテーション設定プロファイル（自動生成）
│   ├── magna_sampled.csv     # MagnaTagATune サンプリング結果（自動生成）
│   ├── gtzan_profile.csv     # GTZAN 使用ファイル一覧（自動生成）
│   ├── GTZAN_Dataset/Data/genres_original/  # GTZAN データセット（10ジャンル×100件）
│   ├── TheMagnaTagATuneDataset/  # MagnaTagATune データセット
│   │   ├── annotations_final.csv
│   │   ├── clip_info_final.csv
│   │   └── 001/, 002/, 003/  # 音声ファイル (.mp3)
│   ├── train_sound_*/        # 学習音声ファイル
│   └── test_sound_*/         # テスト音声ファイル
├── checkpoint_fold{N}.pth    # 学習チェックポイント（Fold ごと、自動生成）
├── best_model_fold{N}.pth    # ベストモデル（Fold ごと、自動生成）
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

# その他の依存パッケージ
pip install -r requirements.txt
```

## 使用方法

### 学習の実行

```bash
cd music_labeling
python train.py
```

### CV + 全データ再学習（デフォルト）

`N_FOLDS = 5` の場合:

1. 5-Fold CV でスコアを計測
2. CV サマリー表示後、20秒待機（Ctrl+C で再学習をスキップ可能）
3. 全データで最終モデルを学習（CV の平均ベストエポック数）
4. テスト推論 → `submit.csv` 出力

### CV なし・全データ学習

`config.py` で `N_FOLDS = 0` に設定:

```python
N_FOLDS = 0  # CV スキップ、全データで直接学習
```

全データで `EPOCHS` 回学習し、そのまま推論します。

### 学習の再開

`config.py` で `RESUME_FROM_CHECKPOINT = True` に設定して再実行します。

```python
RESUME_FROM_CHECKPOINT = True
LOAD_OPTIMIZER_STATE = False       # 新実装ではFalse推奨
LOAD_SCHEDULER_STATE = False       # 新実装ではFalse推奨
```

### 最初からやり直す場合

```bash
del checkpoint_fold*.pth best_model_fold*.pth
python train.py
```

または `config.py` で `RESUME_FROM_CHECKPOINT = False` に設定。

### 音源分離結果の WAV 出力

```bash
python export_separated_wav.py
python export_separated_wav.py train_0 train_1
python export_separated_wav.py --channels vocals drums
```

## 設定パラメータ（config.py）

### 基本設定

| パラメータ | デフォルト値 | 説明 |
|---|---|---|
| `SEED` | 42 | 乱数シード |
| `N_MELS` | 128 | メルバンド数 |
| `SR` | 22050 | サンプリングレート |
| `IMG_SIZE` | 600 | リサイズ後の画像サイズ |
| `NUM_CLASSES` | 10 | 分類クラス数 |
| `NUM_CHANNELS` | 3 | 入力チャンネル数（original / accompaniment / vocals） |

### 学習設定

| パラメータ | デフォルト値 | 説明 |
|---|---|---|
| `BATCH_SIZE` | 24 | バッチサイズ |
| `EPOCHS` | 50 | 最大エポック数 |
| `LR` | 3e-4 | 学習率 |
| `WEIGHT_DECAY` | 1.25e-4 | 重み減衰 |
| `N_FOLDS` | 5 | K-Fold 分割数（0 で CV スキップ） |
| `PATIENCE` | 5 | Early Stopping の patience |
| `USE_AMP` | True | Mixed Precision Training |

### 外部データ設定

| パラメータ | デフォルト値 | 説明 |
|---|---|---|
| `USE_MAGNA_DATA` | True | MagnaTagATune データを学習に含めるか |
| `MAGNA_DATASET_DIR` | data/TheMagnaTagATuneDataset | データセットのディレクトリ |
| `MAGNA_SAMPLES_PER_CLASS` | 50 | クラスあたりのサンプル数 |
| `USE_GTZAN_DATA` | True | GTZAN データを学習に含めるか |
| `GTZAN_GENRES_DIR` | data/GTZAN_Dataset/Data/genres_original | GTZAN ジャンルフォルダのディレクトリ |

### 音源分離設定

| パラメータ | デフォルト値 | 説明 |
|---|---|---|
| `SEPARATION_MODEL` | htdemucs | Demucs モデル名 |
| `SEPARATION_CACHE_DIR` | data/separated | 分離結果のキャッシュ先 |

### データオーギュメンテーション設定

#### 音声波形ベース（50% の確率で適用、3ch 同一パラメータ）

| パラメータ | デフォルト値 | 説明 |
|---|---|---|
| `AUGMENT_PITCH_SHIFT` | True | ピッチシフト（±2 半音） |
| `AUGMENT_TIME_STRETCH` | True | タイムストレッチ（0.8〜1.2 倍速） |
| `AUGMENT_ADD_NOISE` | True | ガウシアンノイズ追加（std 0.001〜0.01） |
| `AUGMENT_TIME_SHIFT` | True | タイムシフト（±0.2 秒） |

#### SpecAugment（メルスペクトログラムベース、50% の確率で適用）

| パラメータ | デフォルト値 | 説明 |
|---|---|---|
| `AUGMENT_TIME_MASKING` | True | Time Masking |
| `AUGMENT_FREQUENCY_MASKING` | True | Frequency Masking |

#### バッチ単位（学習ループ内で適用）

| パラメータ | デフォルト値 | 説明 |
|---|---|---|
| `AUGMENT_CUTMIX` | False | CutMix（矩形切り貼り + ラベル混合） |
| `AUGMENT_MIXUP` | False | MixUp（ピクセル線形混合 + ラベル混合） |
| `AUGMENT_CUTOUT` | False | CutOut（矩形ゼロマスク） |

### 出力ファイル

| ファイル | 説明 |
|---|---|
| `checkpoint_fold{N}.pth` | 学習チェックポイント（Fold ごと） |
| `best_model_fold{N}.pth` | ベストモデル（Fold ごと、最終モデルは fold=N_FOLDS） |
| `submit.csv` | テストデータの予測結果（提出用） |
| `train_log.txt` | 学習ログ |
| `data/augmentation_profile.csv` | オーギュメンテーション設定プロファイル |
| `data/magna_sampled.csv` | MagnaTagATune のサンプリング結果 |
| `data/gtzan_profile.csv` | GTZAN 使用ファイル一覧 |
| `data/separated/*.npz` | 音源分離のキャッシュ |
| `data/precomputed_mel.npz` | メルスペクトログラムのキャッシュ |

## トラブルシューティング

### CUDA out of memory

`config.py` の `BATCH_SIZE` を小さくしてください。`IMG_SIZE` を下げることでも対応可能です（例: 600 → 300）。
`USE_AMP = True` が有効になっていることも確認してください。

### 音源分離に時間がかかる

初回実行時のみ Demucs による音源分離が実行されます。分離結果は `data/separated/` にキャッシュされるため、2回目以降はスキップされます。

### チェックポイントの互換性エラー

モデル構造を変更した場合、古いチェックポイントは読み込めません。

```bash
del checkpoint_fold*.pth best_model_fold*.pth
```

で削除してから再学習してください。

## 参考資料

- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Demucs - Music Source Separation](https://github.com/facebookresearch/demucs)
- [librosa Documentation](https://librosa.org/doc/latest/index.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
