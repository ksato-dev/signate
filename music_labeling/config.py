"""
設定ファイル - 音楽ジャンル分類プロジェクト
"""

# ==============================================================================
# 基本設定
# ==============================================================================
SEED = 42

# ==============================================================================
# 音声処理設定
# ==============================================================================
N_MELS = 128              # メルバンド数
SR = 22050                # サンプリングレート

# ==============================================================================
# モデル設定
# ==============================================================================
IMG_SIZE = 600            # EfficientNet の入力サイズ
NUM_CLASSES = 10          # 分類クラス数
NUM_CHANNELS = 3          # 入力チャンネル数（original / accompaniment / vocals）

# ==============================================================================
# 学習設定
# ==============================================================================
BATCH_SIZE = 24           # バッチサイズ（RTX3090 (24GB) なら16-24が可能。メモリ不足時は下げる）
EPOCHS = 50               # 最大エポック数
LR = 2.5e-4                 # 学習率（バッチサイズ増加に伴い学習率も調整）
# LR = 0.000154                 # 学習率（バッチサイズ増加に伴い学習率も調整）
WEIGHT_DECAY = 1e-4       # 重み減衰
N_FOLDS = 5               # K-Fold 交差検証の分割数（0 で CV スキップ、全データで直接学習）
# N_FOLDS = 0               # K-Fold 交差検証の分割数（0 で CV スキップ、全データで直接学習）

# ==============================================================================
# チェックポイント設定
# ==============================================================================
RESUME_FROM_CHECKPOINT = False      # チェックポイントから再開するか
LOAD_OPTIMIZER_STATE = False       # 最適化器の状態も読み込むか（新実装ではFalse推奨）
LOAD_SCHEDULER_STATE = False       # スケジューラの状態も読み込むか（新実装ではFalse推奨）

# ==============================================================================
# データオーギュメンテーション設定
# ==============================================================================
# 8パターン方式: 元音声 + 7種類の拡張（含 vocal_drop）をランダムに1つ選択
# augment=True 時、毎回 random.choice で1パターンを適用
AUDIO_DURATION = 30               # 全パターン共通: fix_length の目標秒数

# Pitch Shift Up（+半音）
PITCH_UP_MIN = 2
PITCH_UP_MAX = 5

# Pitch Shift Down（-半音）
PITCH_DOWN_MIN = -5
PITCH_DOWN_MAX = -1

# Time Stretch Slow（遅くする）
STRETCH_SLOW_MIN = 0.7
STRETCH_SLOW_MAX = 0.9

# Time Stretch Fast（速くする）
STRETCH_FAST_MIN = 1.1
STRETCH_FAST_MAX = 1.4

# Adding Noise（ガウシアンノイズ）
NOISE_LEVEL_MIN = 0.005
NOISE_LEVEL_MAX = 0.015

# Time Shift（循環シフト: 音声長の割合）
ROLL_FRAC_MIN = 0.1
ROLL_FRAC_MAX = 0.5

# ボーカル除去オーギュメンテーション（ジャンル別確率で伴奏のみにする）
AUGMENT_VOCAL_DROP = True            # ボーカル抜きオーギュメンテーションを適用するか
VOCAL_INTEGRAL_THRESHOLD = 2500      # ボーカル積分値の閾値（これ以下を「低ボーカル」とみなす）
VOCAL_DROP_TARGET_RATIO = 0.5        # ジャンルごとの低ボーカル割合の目標値

# SpecAugment（メルスペクトログラムベース）
AUGMENT_TIME_MASKING = True       # Time Maskingを適用するか
AUGMENT_FREQUENCY_MASKING = True  # Frequency Maskingを適用するか

# 画像系オーギュメンテーション（バッチ単位・学習ループ内で適用）
AUGMENT_CUTMIX = False            # CutMix を適用するか
AUGMENT_CUTMIX_PROB = 0.5        # CutMix を適用する確率
AUGMENT_CUTMIX_ALPHA = 1.0       # CutMix の Beta(alpha, alpha) の alpha

AUGMENT_MIXUP = False             # MixUp を適用するか
AUGMENT_MIXUP_PROB = 0.5        # MixUp を適用する確率
AUGMENT_MIXUP_ALPHA = 0.2        # MixUp の Beta(alpha, alpha) の alpha

AUGMENT_CUTOUT = False            # CutOut（矩形マスクでゼロ化）を適用するか
AUGMENT_CUTOUT_PROB = 0.5       # CutOut を適用する確率
AUGMENT_CUTOUT_N_HOLES = 1      # マスクする矩形の数
AUGMENT_CUTOUT_RATIO = 0.15    # マスクの辺の長さ（画像辺に対する比率, 0〜1）

# ==============================================================================
# MagnaTagATune 外部データセット設定
# ==============================================================================
USE_MAGNA_DATA = True                          # MagnaTagATune データを学習に含めるか
MAGNA_DATASET_DIR = 'data/TheMagnaTagATuneDataset'
MAGNA_ANNOTATIONS_FILE = 'data/TheMagnaTagATuneDataset/annotations_final.csv'
MAGNA_SAMPLES_PER_CLASS = 52                   # クラスあたりのサンプル数（均等サンプリング）

# ==============================================================================
# GTZAN 外部データセット設定
# ==============================================================================
USE_GTZAN_DATA = True                          # GTZAN データを学習に含めるか
GTZAN_GENRES_DIR = 'data/GTZAN_Dataset/Data/genres_original'

# ==============================================================================
# FMA (Free Music Archive) 外部データセット設定
# ==============================================================================
USE_FMA_DATA = True                            # FMA データを学習に含めるか
FMA_BASE_DIR = 'data/FMA-Free_Music_Archive-Small&Medium'
FMA_TRACKS_CSV = 'data/FMA-Free_Music_Archive-Small&Medium/fma_metadata/tracks.csv'
FMA_SUBSETS = ['small', 'medium']              # 使用するサブセット
FMA_MIN_DURATION = 30                          # 最小秒数（これ以上の曲のみ使用、先頭30秒を切り出す）
FMA_SAMPLES_PER_CLASS = 365                    # クラスあたりのサンプル数（均等サンプリング）

# ==============================================================================
# 音源分離設定（Demucs）
# ==============================================================================
SEPARATION_MODEL = 'htdemucs'         # Demucsモデル名（htdemucs, htdemucs_ft, mdx_extra等）
SEPARATION_CACHE_DIR = 'data/separated'  # 分離結果のキャッシュディレクトリ

# ==============================================================================
# ファイルパス設定
# ==============================================================================
MEL_CACHE_FILE = 'data/precomputed_mel.npz'  # 事前計算メルスペクトログラムキャッシュ（ロード短縮用）
CHECKPOINT_FILE = 'checkpoint_fold{}.pth'    # {} に Fold 番号が入る
BEST_MODEL_FILE = 'best_model_fold{}.pth'    # {} に Fold 番号が入る
SUBMIT_FILE = 'submit.csv'
LOG_FILE = 'train_log.txt'          # 学習ログの出力先

# ==============================================================================
# Mixed Precision
# ==============================================================================
USE_AMP = True                # Mixed Precision Training（VRAM約40-50%削減）

# ==============================================================================
# Early Stopping設定
# ==============================================================================
PATIENCE = 8             # Early Stoppingのpatience
