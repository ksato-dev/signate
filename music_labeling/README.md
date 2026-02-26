# 音楽ジャンル分類 - EfficientNet B7

音楽音声ファイルを10種類のジャンルに分類する深層学習モデルです。
Demucs で音源を分離し、3チャンネルメルスペクトログラム（原音 / 伴奏 / ボーカル）を EfficientNet B7 で Fine-tuning します。

## プロジェクト概要

- **タスク**: 音楽ジャンル分類（10クラス）
- **モデル**: EfficientNet B7 (ImageNet事前学習済み)
- **音源分離**: Demucs (htdemucs) — 4ステム分離 → 3チャンネルに変換
- **フレームワーク**: PyTorch
- **入力**: 3チャンネルメルスペクトログラム（Original / Accompaniment / Vocals → 600x600にリサイズ）
- **出力**: 10クラス分類（blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock）

## モデルアーキテクチャ

```
Input: (B, 3, H, W)   ← 3ch メルスペクトログラム（original / accompaniment / vocals）
        ↓
EfficientNet B7（features[:4] 凍結）
        ↓
Classifier:
  Dropout(0.5) → Linear(2560, 512) → ReLU → Dropout(0.3) → Linear(512, 10)
        ↓
Output logits
```

3チャンネル入力は EfficientNet B7 の標準 RGB 入力とそのまま一致するため、Conv 層の改変は不要です。

### 設計ポイント

| 要素 | 内容 |
|---|---|
| EfficientNet B7 | ImageNet 事前学習済みの CNN。`features[:4]` を凍結して過学習を防止 |
| 3ch 入力 | Original / Accompaniment / Vocals を個別チャンネルとして入力 |
| カスタム分類ヘッド | Dropout(0.5) + Linear(2560→512) + ReLU + Dropout(0.3) + Linear(512→10) |
| Mixed Precision (AMP) | float16 で計算し VRAM 使用量を約40-50%削減 |

## 処理フロー

```
1. 音源分離 (Demucs 4stem → 3ch: original / accompaniment / vocals)
        ↓
2. メルスペクトログラム抽出 (128 mel bands)
        ↓
3. 前処理 (パディング → チャンネルごと標準化 → 600x600 リサイズ)
        ↓
4. K-Fold 交差検証 (StratifiedKFold, N_FOLDS=4)
   ├── 元データのみで分割し、外部データは学習に全量追加
   ├── Fold ごとにモデル学習 + Early Stopping (patience=8)
   └── CV スコアサマリー + ジャンル別エラーログ出力
        ↓  ← Ctrl+C で中止可能 (20秒待機)
5. 全データで最終モデルを再学習 (5 シード・アンサンブル, CV 平均ベストエポック数)
        ↓
6. テスト推論 → 5 モデルの softmax 平均 → argmax → submit.csv 出力
```

`N_FOLDS = 0` に設定すると CV をスキップし、全データで直接学習します。

## 最終モデルの内訳

### アンサンブル構成

5つの異なるシードで同一アーキテクチャのモデルを学習し、推論時に softmax 平均でアンサンブルします。

| # | シード | モデルファイル |
|---|--------|----------------|
| 1 | 42     | `best_model_foldseed42.pth` |
| 2 | 123    | `best_model_foldseed123.pth` |
| 3 | 456    | `best_model_foldseed456.pth` |
| 4 | 789    | `best_model_foldseed789.pth` |
| 5 | 1024   | `best_model_foldseed1024.pth` |

### 学習データ構成

| データソース | 有効 | 件数/クラス（基本） | 備考 |
|---|---|---|---|
| コンペ学習データ | ○ | — | `train_master.csv` に基づく全件使用 |
| MagnaTagATune | ○ | 52 | `annotations_final.csv` のジャンルタグで均等サンプリング |
| GTZAN | ✕ | — | `USE_GTZAN_DATA = False` で無効 |
| FMA | ✕ | — | `USE_FMA_DATA = False` で無効 |

### 学習設定サマリ

| 項目 | 値 |
|---|---|
| バックボーン | EfficientNet B7 (ImageNet V1) |
| 凍結層 | `features[:4]` |
| 分類ヘッド | Dropout(0.5) → FC(2560→512) → ReLU → Dropout(0.3) → FC(512→10) |
| 入力サイズ | 600 × 600 |
| バッチサイズ | 12 |
| 最大エポック数 | 50 |
| 学習率 | 1.25e-4 (AdamW) |
| 重み減衰 | 1e-4 |
| スケジューラ | CosineAnnealingLR (eta_min=1e-6) |
| CV | 4-Fold StratifiedKFold |
| Early Stopping | patience=8 |
| 最終学習エポック数 | CV 平均ベストエポック（自動決定） |
| アンサンブル | 5 シード softmax 平均 |
| MagnaTagATune | 52 件/クラス（基本） |
| オーバーサンプリング | Accuracy ベース (無効) + ジャンル別倍率オーバーライド |

### データオーギュメンテーション

8パターンからランダムに1つを選択し、3チャンネル（原音/伴奏/ボーカル）に同一パラメータで適用します。

| パターン | パラメータ |
|---|---|
| original | 変換なし（元音声そのまま） |
| pitch_up | +2〜+5 半音 |
| pitch_down | -5〜-1 半音 |
| stretch_slow | 0.7〜0.9 倍速 |
| stretch_fast | 1.1〜1.4 倍速 |
| noise | ガウシアンノイズ σ=0.005〜0.015 |
| time_shift | 循環シフト 10%〜50% |
| vocal_drop | ボーカル除去（伴奏のみにする） |

**SpecAugment（メルスペクトログラムベース）**: Time Masking + Frequency Masking（各50%確率）

**画像系オーギュメンテーション**: CutMix / MixUp / CutOut はすべて無効

**ボーカル抜きオーギュメンテーション**: `AUGMENT_VOCAL_DROP = False`（無効）

### 外部データ・オーバーサンプリング設定

MagnaTagATune のサンプリングに Accuracy ベースのジャンル別オーバーサンプリングを適用可能です（現在は `OVERSAMPLE_BY_ACCURACY = False`）。

ジャンル別倍率オーバーライド（`OVERSAMPLE_GENRE_MULT_OVERRIDE`）:

| ジャンル | 倍率 | 実効件数 |
|---|---|---|
| blues | 1.5x | 78 |
| country | 1.6x | 83 |
| hiphop | 1.7x | 88 |
| pop | 2.0x | 104 |
| rock | 2.0x | 104 |
| その他 | 1.0x | 52 |

不足分は FMA からのバックフィルおよび残余プールからの補填が可能です。

## 外部データ: MagnaTagATune

MagnaTagATune データセットの音声ファイルを、ジャンルタグに基づいて学習データに追加できます。

- `annotations_final.csv` の 10 ジャンルカラムを `label_master.csv` と照合
- 複数ジャンルタグを持つクリップは最も件数が少ないジャンルに優先割り当て
- クラスごとに均等サンプリング（基本 52 件/クラス、オーバーサンプリング設定で増量可能）
- `USE_MAGNA_DATA = False` で無効化可能
- サンプリング結果は `data/magna_sampled.csv` に出力

## 外部データ: GTZAN

GTZAN Dataset (`genres_original`) の音声ファイルを学習データに追加できます。

- 10ジャンル × 100ファイル = 1000件（完全均等バランス）
- フォルダ名が `label_master.csv` のジャンル名と完全一致するため全件そのまま使用
- `USE_GTZAN_DATA = False` で無効化可能（現在は無効）
- プロファイルは `data/gtzan_profile.csv` に出力

## 外部データ: FMA (Free Music Archive)

FMA Small / Medium サブセットからジャンルラベル付きサンプルを均等サンプリングして追加できます。

- `tracks.csv` の `genre_top` / `genres_all` からラベルを割り当て
- disco, metal, reggae は `genres_all` の genre_id で直接マッチング
- クラスごとに基本 74 件/クラスで均等サンプリング
- MagnaTagATune の不足分を FMA で自動補填（バックフィル）
- `USE_FMA_DATA = False` で無効化可能（現在は無効）
- サンプリング結果は `data/fma_sampled.csv` に出力

## ディレクトリ構造

```
music_labeling/
├── train.py                  # メイン学習スクリプト
├── predict_submit.py         # 推論・提出ファイル作成
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
│   ├── fma_sampled.csv       # FMA サンプリング結果（自動生成）
│   ├── GTZAN_Dataset/Data/genres_original/  # GTZAN データセット（10ジャンル×100件）
│   ├── TheMagnaTagATuneDataset/  # MagnaTagATune データセット
│   │   ├── annotations_final.csv
│   │   ├── clip_info_final.csv
│   │   └── 001/, 002/, 003/  # 音声ファイル (.mp3)
│   ├── FMA-Free_Music_Archive-Small&Medium/  # FMA データセット
│   │   ├── fma_metadata/tracks.csv
│   │   ├── fma_small/fma_small/
│   │   └── fma_medium/fma_medium/
│   ├── train_sound_*/        # 学習音声ファイル
│   └── test_sound_*/         # テスト音声ファイル
├── checkpoint_fold{N}.pth    # 学習チェックポイント（Fold ごと、自動生成）
├── best_model_fold{N}.pth    # ベストモデル（Fold ごと）
├── best_model_foldseed{S}.pth  # 最終アンサンブルモデル（シードごと）
├── cv_genre_errors.log       # CV ジャンル別エラー分析ログ（自動生成）
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

### CV + アンサンブル学習（デフォルト）

`N_FOLDS = 4`, `ENSEMBLE_SEEDS = [42, 123, 456, 789, 1024]` の場合:

1. 4-Fold CV でスコアを計測（ジャンル別エラーログを `cv_genre_errors.log` に出力）
2. CV サマリー表示後、20秒待機（Ctrl+C で再学習をスキップ可能）
3. 全データで最終モデルを5シードで学習（CV の平均ベストエポック数）
4. テスト推論 → 5モデルの softmax 平均 → `submit.csv` 出力

### CV なし・全データ学習

`config.py` で `N_FOLDS = 0` に設定:

```python
N_FOLDS = 0  # CV スキップ、全データで直接学習
```

全データで `EPOCHS` 回（または `EPOCHS_TO_FORCE_FINISH` 回）学習し、そのまま推論します。

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
| `ENSEMBLE_SEEDS` | [42, 123, 456, 789, 1024] | アンサンブル用シードリスト |
| `N_MELS` | 128 | メルバンド数 |
| `SR` | 22050 | サンプリングレート |
| `IMG_SIZE` | 600 | リサイズ後の画像サイズ |
| `NUM_CLASSES` | 10 | 分類クラス数 |
| `NUM_CHANNELS` | 3 | 入力チャンネル数（original / accompaniment / vocals） |

### 学習設定

| パラメータ | デフォルト値 | 説明 |
|---|---|---|
| `BATCH_SIZE` | 12 | バッチサイズ（B7 は VRAM 消費が大きいため小さめ） |
| `EPOCHS` | 50 | 最大エポック数 |
| `LR` | 1.25e-4 | 学習率 (AdamW) |
| `WEIGHT_DECAY` | 1e-4 | 重み減衰 |
| `N_FOLDS` | 4 | K-Fold 分割数（0 で CV スキップ） |
| `PATIENCE` | 8 | Early Stopping の patience |
| `USE_AMP` | True | Mixed Precision Training |
| `EPOCHS_TO_FORCE_FINISH` | None | N_FOLDS=0 時のエポック上限（None で EPOCHS まで実行） |

### 外部データ設定

| パラメータ | デフォルト値 | 説明 |
|---|---|---|
| `USE_MAGNA_DATA` | True | MagnaTagATune データを学習に含めるか |
| `MAGNA_DATASET_DIR` | data/TheMagnaTagATuneDataset | データセットのディレクトリ |
| `MAGNA_SAMPLES_PER_CLASS` | 52 | クラスあたりの基本サンプル数 |
| `USE_GTZAN_DATA` | False | GTZAN データを学習に含めるか |
| `GTZAN_GENRES_DIR` | data/GTZAN_Dataset/Data/genres_original | GTZAN ジャンルフォルダのディレクトリ |
| `USE_FMA_DATA` | False | FMA データを学習に含めるか |
| `FMA_BASE_DIR` | data/FMA-Free_Music_Archive-Small&Medium | FMA データセットの基本ディレクトリ |
| `FMA_SUBSETS` | ['small', 'medium'] | 使用する FMA サブセット |
| `FMA_MIN_DURATION` | 30 | 最小秒数フィルタ |
| `FMA_SAMPLES_PER_CLASS` | 74 | クラスあたりの基本サンプル数 |

### オーバーサンプリング設定

| パラメータ | デフォルト値 | 説明 |
|---|---|---|
| `OVERSAMPLE_BY_ACCURACY` | False | Accuracy ベースの外部データ増量を有効化 |
| `OVERSAMPLE_ACC_THRESHOLD` | 80.0 | この Accuracy(%) 以下のジャンルを増量対象にする |
| `OVERSAMPLE_MAX_MULTIPLIER` | 2.0 | 最大倍率 |
| `OVERSAMPLE_GENRE_MULT_OVERRIDE` | {blues:1.5, country:1.6, hiphop:1.7, pop:2.0, rock:2.0} | ジャンル別倍率の直接指定 |

### 音源分離設定

| パラメータ | デフォルト値 | 説明 |
|---|---|---|
| `SEPARATION_MODEL` | htdemucs | Demucs モデル名 |
| `SEPARATION_CACHE_DIR` | data/separated | 分離結果のキャッシュ先 |

### データオーギュメンテーション設定

#### 音声波形ベース（8パターンからランダム1つ選択、3ch 同一パラメータ）

| パラメータ | デフォルト値 | 説明 |
|---|---|---|
| `AUDIO_DURATION` | 30 | 全パターン共通の目標秒数 |
| `PITCH_UP_MIN` / `MAX` | 2 / 5 | ピッチアップ（半音） |
| `PITCH_DOWN_MIN` / `MAX` | -5 / -1 | ピッチダウン（半音） |
| `STRETCH_SLOW_MIN` / `MAX` | 0.7 / 0.9 | テンポ遅延 |
| `STRETCH_FAST_MIN` / `MAX` | 1.1 / 1.4 | テンポ加速 |
| `NOISE_LEVEL_MIN` / `MAX` | 0.005 / 0.015 | ガウシアンノイズ σ |
| `ROLL_FRAC_MIN` / `MAX` | 0.1 / 0.5 | 循環シフト割合 |
| `AUGMENT_VOCAL_DROP` | False | ボーカル抜きオーギュメンテーション |

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
| `best_model_fold{N}.pth` | ベストモデル（Fold ごと） |
| `best_model_foldseed{S}.pth` | 最終アンサンブルモデル（シードごと、5個） |
| `submit.csv` | テストデータの予測結果（提出用） |
| `cv_genre_errors.log` | CV ジャンル別エラー分析ログ（混同行列・誤分類ファイル一覧） |
| `train_log.txt` | 学習ログ |
| `data/augmentation_profile.csv` | オーギュメンテーション設定プロファイル |
| `data/magna_sampled.csv` | MagnaTagATune のサンプリング結果 |
| `data/gtzan_profile.csv` | GTZAN 使用ファイル一覧 |
| `data/fma_sampled.csv` | FMA サンプリング結果 |
| `data/separated/*.npz` | 音源分離のキャッシュ（4ステム） |
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
