"""
音楽ジャンル分類 - EfficientNet B3 (PyTorch)
音源分離（Demucs）で原音・伴奏・ボーカルに分離し、
3チャンネルメルスペクトログラムで Fine-tuning
"""

import os
import glob
import signal
import sys
import numpy as np
import pandas as pd
import librosa
from natsort import natsorted
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import warnings
warnings.simplefilter('ignore')

# 設定ファイルからインポート
import config

# 設定をconfigから取得
SEED = config.SEED
N_MELS = config.N_MELS
SR = config.SR
IMG_SIZE = config.IMG_SIZE
BATCH_SIZE = config.BATCH_SIZE
EPOCHS = config.EPOCHS
LR = config.LR
WEIGHT_DECAY = config.WEIGHT_DECAY
NUM_CLASSES = config.NUM_CLASSES
TEST_SIZE = config.TEST_SIZE
RESUME_FROM_CHECKPOINT = config.RESUME_FROM_CHECKPOINT
LOAD_OPTIMIZER_STATE = config.LOAD_OPTIMIZER_STATE
LOAD_SCHEDULER_STATE = config.LOAD_SCHEDULER_STATE
CHECKPOINT_FILE = config.CHECKPOINT_FILE
BEST_MODEL_FILE = config.BEST_MODEL_FILE
SUBMIT_FILE = config.SUBMIT_FILE
PATIENCE = config.PATIENCE
SEPARATION_CACHE_DIR = config.SEPARATION_CACHE_DIR
SEPARATION_MODEL = config.SEPARATION_MODEL

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Device: {DEVICE}")

# ==============================================================================
# 4ステム名 / キャッシュ判定 / 3ch 変換ヘルパー
# ==============================================================================
STEM_NAMES_4 = ['vocals', 'drums', 'bass', 'other']


def _is_4stem_cache(cache_path):
    """キャッシュが4ステム形式かどうかを判定"""
    if not os.path.exists(cache_path):
        return False
    try:
        data = np.load(cache_path)
        result = 'drums' in data
        data.close()
        return result
    except Exception:
        return False


def load_3ch_from_cache(data):
    """任意のキャッシュ形式 (3ch or 4stem) から (original, accompaniment, vocals) を返す

    4stem (vocals, drums, bass, other; optional original):
        original = data['original'] があればそれ、なければ全ステムの和
        accompaniment = drums + bass + other
    3ch (original, accompaniment, vocals):
        そのまま返す
    """
    if 'drums' in data:
        vocals = data['vocals']
        drums = data['drums']
        bass = data['bass']
        other = data['other']
        accompaniment = drums + bass + other
        original = data['original'] if 'original' in data else (vocals + accompaniment)
        return original, accompaniment, vocals
    else:
        return data['original'], data['accompaniment'], data['vocals']


# ==============================================================================
# 音源分離（Demucs）4ステム
# ==============================================================================
def separate_audio_cached(file_path, demucs_model, cache_dir=SEPARATION_CACHE_DIR, sr=SR,
                          device=DEVICE):
    """音声ファイルをDemucsで4ステム（vocals/drums/bass/other）に分離しキャッシュする

    旧3chキャッシュ（original/accompaniment/vocals）が見つかった場合は
    自動で再分離して4ステム形式に変換する。

    Args:
        file_path: 音声ファイルパス
        demucs_model: demucs.pretrained.get_model() で取得したモデル
        cache_dir: キャッシュディレクトリ
        sr: 出力サンプリングレート
        device: 推論デバイス

    Returns:
        tuple: (vocals, drums, bass, other) の各波形 (numpy array)
    """
    from demucs.apply import apply_model

    basename = os.path.splitext(os.path.basename(file_path))[0]
    cache_path = os.path.join(cache_dir, f"{basename}.npz")

    # 4ステムキャッシュが存在すればロード
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        if 'drums' in data:
            stems = tuple(data[n] for n in STEM_NAMES_4)
            data.close()
            return stems
        data.close()
        print(f"  旧3chキャッシュを検出: {basename} → 4stem で再分離します")
        os.remove(cache_path)

    os.makedirs(cache_dir, exist_ok=True)

    # 原音を目標サンプリングレートで読み込み
    original, _ = librosa.load(file_path, sr=sr, mono=True)

    # Demucsのネイティブサンプリングレートで読み込み
    demucs_sr = demucs_model.samplerate
    y_demucs, _ = librosa.load(file_path, sr=demucs_sr, mono=True)

    # Demucsはステレオ入力を想定 → モノラルを疑似ステレオに
    waveform = torch.FloatTensor(y_demucs).unsqueeze(0).repeat(2, 1)  # (2, time)
    mix = waveform.unsqueeze(0)  # (1, 2, time)

    with torch.no_grad():
        sources = apply_model(demucs_model, mix, device=device)

    sources = sources.squeeze(0)  # (n_sources, 2, time)
    source_names = demucs_model.sources  # ['drums', 'bass', 'other', 'vocals']

    # 各ステムを抽出（ステレオ→モノラル）してリサンプル
    stems = {}
    for name in STEM_NAMES_4:
        idx = source_names.index(name)
        raw = sources[idx].mean(dim=0).cpu().numpy()
        stems[name] = librosa.resample(raw, orig_sr=demucs_sr, target_sr=sr) if demucs_sr != sr else raw

    # 長さを揃える（原音含む）
    min_len = min(len(original), *(len(v) for v in stems.values()))
    original = original[:min_len]
    for name in stems:
        stems[name] = stems[name][:min_len]

    # 4ステム + original（元の音声ファイル）でキャッシュ保存
    np.savez_compressed(cache_path, original=original, **stems)

    return tuple(stems[n] for n in STEM_NAMES_4)


def batch_separate(file_list, demucs_model, cache_dir=SEPARATION_CACHE_DIR, sr=SR,
                   device=DEVICE):
    """ファイルリスト全体を4ステム分離（旧3chキャッシュも自動で再分離）"""
    cached_4stem = sum(
        1 for f in file_list
        if _is_4stem_cache(
            os.path.join(cache_dir,
                         f"{os.path.splitext(os.path.basename(f))[0]}.npz"))
    )
    if cached_4stem == len(file_list):
        print(f"  全{len(file_list)}ファイルが4ステムキャッシュ済み")
        return
    print(f"  4ステムキャッシュ済み: {cached_4stem}/{len(file_list)}, "
          f"残り{len(file_list) - cached_4stem}ファイルを分離中...")
    for f in tqdm(file_list, desc="  音源分離"):
        separate_audio_cached(f, demucs_model, cache_dir, sr, device)


# ==============================================================================
# メルスペクトログラム抽出
# ==============================================================================
def extract_melspectrogram(y, sr=SR, n_mels=None):
    """音声波形からメルスペクトログラム(dB)を抽出

    Args:
        y: 音声波形データ
        sr: サンプリングレート
        n_mels: メルバンド数。Noneの場合はlibrosaのデフォルト値(128)を使用
    """
    kwargs = {} if n_mels is None else {'n_mels': n_mels}
    mel = librosa.feature.melspectrogram(y=y, sr=sr, **kwargs)
    mel_db = librosa.amplitude_to_db(mel, ref=np.max)
    return mel_db


# ==============================================================================
# オーギュメンテーション（3チャンネル共通パラメータ方式）
# ==============================================================================
def generate_augment_params(sr=SR):
    """オーギュメンテーションパラメータを事前に生成

    3チャンネル（原音・伴奏・ボーカル）に同じオーギュメンテーションを
    適用するため、パラメータを事前に生成する。
    """
    params = {}

    if config.AUGMENT_PITCH_SHIFT and np.random.random() < 0.5:
        params['pitch_n_steps'] = np.random.uniform(
            config.PITCH_SHIFT_MIN, config.PITCH_SHIFT_MAX)

    if config.AUGMENT_TIME_STRETCH and np.random.random() < 0.5:
        params['stretch_rate'] = np.random.uniform(
            config.TIME_STRETCH_MIN, config.TIME_STRETCH_MAX)

    if config.AUGMENT_ADD_NOISE and np.random.random() < 0.5:
        params['noise_factor'] = np.random.uniform(
            config.NOISE_FACTOR_MIN, config.NOISE_FACTOR_MAX)

    if config.AUGMENT_TIME_SHIFT and np.random.random() < 0.5:
        shift_max = int(sr * config.TIME_SHIFT_MAX_SEC)
        params['time_shift'] = np.random.randint(-shift_max, shift_max + 1)

    return params


def apply_augment(y, params, sr=SR, noise_seed=None):
    """生成済みパラメータでオーギュメンテーションを適用

    Args:
        y: 音声波形
        params: generate_augment_params() で生成したパラメータ
        sr: サンプリングレート
        noise_seed: ノイズ用シード（3チャンネルに同じノイズを適用する場合に使用）
    """
    if 'pitch_n_steps' in params:
        y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=params['pitch_n_steps'])

    if 'stretch_rate' in params:
        y = librosa.effects.time_stretch(y=y, rate=params['stretch_rate'])

    if 'noise_factor' in params:
        if noise_seed is not None:
            rng = np.random.RandomState(noise_seed)
            noise = rng.randn(len(y)) * params['noise_factor']
        else:
            noise = np.random.randn(len(y)) * params['noise_factor']
        y = y + noise

    if 'time_shift' in params:
        shift = params['time_shift']
        if shift > 0:
            y = np.pad(y, (shift, 0), mode='constant')[:-shift]
        elif shift < 0:
            y = np.pad(y, (0, -shift), mode='constant')[-shift:]

    return y


# ==============================================================================
# 3チャンネル メルスペクトログラム読み込み
# ==============================================================================
def load_and_pad_3ch(file_list, cache_dir=SEPARATION_CACHE_DIR, sr=SR, n_mels=N_MELS):
    """分離済み音声から3チャンネルメルスペクトログラムを抽出し、パディングして統一サイズにする

    チャンネル構成:
        ch0: 原音（Original）
        ch1: 伴奏（Accompaniment）
        ch2: ボーカル（Vocals）

    Returns:
        spectrograms: (N, 3, n_mels, max_frames) の numpy 配列
        max_frames: 最大フレーム数
    """
    all_mels = []

    for f in tqdm(file_list, desc="  メルスペクトログラム抽出"):
        basename = os.path.splitext(os.path.basename(f))[0]
        cache_path = os.path.join(cache_dir, f"{basename}.npz")
        data = np.load(cache_path)

        original, accompaniment, vocals = load_3ch_from_cache(data)

        mel_orig = extract_melspectrogram(original, sr=sr, n_mels=n_mels)
        mel_acc = extract_melspectrogram(accompaniment, sr=sr, n_mels=n_mels)
        mel_voc = extract_melspectrogram(vocals, sr=sr, n_mels=n_mels)

        # フレーム数を揃える（リサンプル由来のわずかなずれを補正）
        min_frames = min(mel_orig.shape[1], mel_acc.shape[1], mel_voc.shape[1])
        mel_orig = mel_orig[:, :min_frames]
        mel_acc = mel_acc[:, :min_frames]
        mel_voc = mel_voc[:, :min_frames]

        # (3, n_mels, time_frames)
        mel_3ch = np.stack([mel_orig, mel_acc, mel_voc], axis=0)
        all_mels.append(mel_3ch)

    # 最大フレーム数に合わせてパディング
    max_frames = max(m.shape[2] for m in all_mels)
    padded = []
    for m in all_mels:
        pad_width = max_frames - m.shape[2]
        if pad_width > 0:
            p = np.pad(m, ((0, 0), (0, 0), (0, pad_width)),
                       mode='constant', constant_values=m.min())
        else:
            p = m
        padded.append(p)

    return np.array(padded), max_frames


# ==============================================================================
# 音源分離の実行
# ==============================================================================
print("=== データファイルの読み込み ===")
train_master = pd.read_csv('data/train_master.csv', index_col=0)
label_master = pd.read_csv('data/label_master.csv')

train_files = natsorted(glob.glob('data/train_sound_*/train_sound_*/train_*.au'))
test_files = natsorted(glob.glob('data/test_sound_*/test_sound_*/test_*.au'))
print(f"学習ファイル数: {len(train_files)}")
print(f"テストファイル数: {len(test_files)}")

print("\n=== 音源分離（Demucs） ===")
from demucs.pretrained import get_model as get_demucs_model

print(f"Demucsモデル ({SEPARATION_MODEL}) をロード中...")
demucs_model = get_demucs_model(SEPARATION_MODEL)
demucs_model.to(DEVICE)
print(f"Demucsソース: {demucs_model.sources}")
print(f"Demucsサンプルレート: {demucs_model.samplerate} Hz")

print("\n学習データの音源分離:")
batch_separate(train_files, demucs_model, SEPARATION_CACHE_DIR, SR, DEVICE)

print("\nテストデータの音源分離:")
batch_separate(test_files, demucs_model, SEPARATION_CACHE_DIR, SR, DEVICE)

# Demucsモデルを解放してGPUメモリを確保
del demucs_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("\nDemucsモデルを解放しました")

# ==============================================================================
# 3チャンネル メルスペクトログラム抽出
# ==============================================================================
print("\n=== 3チャンネル メルスペクトログラム抽出 ===")
mel_freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=0.0, fmax=SR / 2.0)
print(f"メル周波数バンド: {N_MELS}バンド, {mel_freqs[0]:.1f} - {mel_freqs[-1]:.1f} Hz")

print("\n学習データ:")
X_all, max_frames_train = load_and_pad_3ch(train_files, SEPARATION_CACHE_DIR, SR, N_MELS)
y_all = train_master['label_id'].values
print(f"X_all shape: {X_all.shape}")  # (N, 3, n_mels, max_frames)
print(f"y_all shape: {y_all.shape}")

print("\nテストデータ:")
X_test_raw, max_frames_test = load_and_pad_3ch(test_files, SEPARATION_CACHE_DIR, SR, N_MELS)

# テストデータのフレーム数を学習データに合わせる
if max_frames_test > max_frames_train:
    X_test_raw = X_test_raw[:, :, :, :max_frames_train]
elif max_frames_test < max_frames_train:
    pad_width = max_frames_train - max_frames_test
    X_test_raw = np.pad(X_test_raw,
                        ((0, 0), (0, 0), (0, 0), (0, pad_width)),
                        mode='constant', constant_values=X_test_raw.min())
print(f"X_test shape: {X_test_raw.shape}")


# ==============================================================================
# Dataset（3チャンネル対応）
# ==============================================================================
class MelSpectrogramDataset(Dataset):
    """3チャンネル（原音・伴奏・ボーカル）メルスペクトログラム Dataset

    ch0 = Original（原音）
    ch1 = Accompaniment（伴奏）
    ch2 = Vocals（ボーカル）
    """

    def __init__(self, X, file_list=None, y=None, img_size=IMG_SIZE,
                 augment=False, sr=SR, n_mels=N_MELS, max_frames=None,
                 cache_dir=SEPARATION_CACHE_DIR, vocal_integrals=None, genre_drop_prob=None):
        self.X = X  # (N, 3, n_mels, time_frames)
        self.file_list = file_list
        self.y = y
        self.img_size = img_size
        self.augment = augment
        self.sr = sr
        self.n_mels = n_mels
        self.max_frames = max_frames
        self.cache_dir = cache_dir
        self.vocal_integrals = vocal_integrals  # 訓練用: サンプルごとのボーカル積分
        self.genre_drop_prob = genre_drop_prob  # ジャンルごとのボーカル抜き適用確率

        # 正規化パラメータ（チャンネルごとに計算）
        self.ch_means = np.array([X[:, ch].mean() for ch in range(3)])  # (3,)
        self.ch_stds = np.array([X[:, ch].std() for ch in range(3)])    # (3,)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.augment and self.file_list is not None:
            # === オーギュメンテーションあり ===
            # キャッシュから分離済み波形を読み込み
            basename = os.path.splitext(os.path.basename(self.file_list[idx]))[0]
            cache_path = os.path.join(self.cache_dir, f"{basename}.npz")
            data = np.load(cache_path)

            original, accompaniment, vocals = load_3ch_from_cache(data)
            original = original.copy()
            accompaniment = accompaniment.copy()
            vocals = vocals.copy()

            # ボーカル抜きオーギュメンテーション: 積分>閾値のサンプルをジャンル別確率で伴奏のみに
            if (self.vocal_integrals is not None and self.genre_drop_prob is not None
                    and self.y is not None):
                th = getattr(config, 'VOCAL_INTEGRAL_THRESHOLD', 2500)
                label_id = int(self.y[idx])
                p = self.genre_drop_prob.get(label_id, 0.0)
                if self.vocal_integrals[idx] > th and p > 0 and np.random.random() < p:
                    vocals = np.zeros_like(vocals)
                    original = accompaniment.copy()

            # 同一パラメータで3チャンネルすべてをオーギュメント
            aug_params = generate_augment_params(sr=self.sr)
            noise_seed = np.random.randint(0, 2**31)

            original = apply_augment(original, aug_params,
                                     sr=self.sr, noise_seed=noise_seed)
            accompaniment = apply_augment(accompaniment, aug_params,
                                          sr=self.sr, noise_seed=noise_seed)
            vocals = apply_augment(vocals, aug_params,
                                   sr=self.sr, noise_seed=noise_seed)

            # メルスペクトログラムに変換
            mel_orig = extract_melspectrogram(original, sr=self.sr, n_mels=self.n_mels)
            mel_acc = extract_melspectrogram(accompaniment, sr=self.sr, n_mels=self.n_mels)
            mel_voc = extract_melspectrogram(vocals, sr=self.sr, n_mels=self.n_mels)

            # フレーム数を揃える
            min_frames = min(mel_orig.shape[1], mel_acc.shape[1], mel_voc.shape[1])
            mel_orig = mel_orig[:, :min_frames]
            mel_acc = mel_acc[:, :min_frames]
            mel_voc = mel_voc[:, :min_frames]

            mel_3ch = np.stack([mel_orig, mel_acc, mel_voc], axis=0)  # (3, n_mels, frames)

            # パディング
            if self.max_frames is not None:
                pad_width = self.max_frames - mel_3ch.shape[2]
                if pad_width > 0:
                    mel_3ch = np.pad(mel_3ch,
                                     ((0, 0), (0, 0), (0, pad_width)),
                                     mode='constant', constant_values=mel_3ch.min())
                elif pad_width < 0:
                    mel_3ch = mel_3ch[:, :, :self.max_frames]
        else:
            # === オーギュメンテーションなし ===
            mel_3ch = self.X[idx].copy()  # (3, n_mels, time_frames)

        # チャンネルごとに標準化
        for ch in range(3):
            mel_3ch[ch] = (mel_3ch[ch] - self.ch_means[ch]) / (self.ch_stds[ch] + 1e-8)

        # SpecAugment（3チャンネルに同じマスクを適用）
        if self.augment:
            # Time masking
            if config.AUGMENT_TIME_MASKING and np.random.random() < 0.5:
                t = mel_3ch.shape[2]
                t_mask = np.random.randint(0, max(1, t // 10))
                t_start = np.random.randint(0, max(1, t - t_mask))
                mel_3ch[:, :, t_start:t_start + t_mask] = 0

            # Frequency masking
            if config.AUGMENT_FREQUENCY_MASKING and np.random.random() < 0.5:
                f = mel_3ch.shape[1]
                f_mask = np.random.randint(0, max(1, f // 10))
                f_start = np.random.randint(0, max(1, f - f_mask))
                mel_3ch[:, f_start:f_start + f_mask, :] = 0

        # Tensor に変換 (3, n_mels, time_frames)
        mel_tensor = torch.FloatTensor(mel_3ch)

        # リサイズ (3, n_mels, time_frames) -> (3, img_size, img_size)
        mel_tensor = torch.nn.functional.interpolate(
            mel_tensor.unsqueeze(0), size=(self.img_size, self.img_size),
            mode='bilinear', align_corners=False
        ).squeeze(0)

        if self.y is not None:
            label = torch.LongTensor([self.y[idx]]).squeeze()
            return mel_tensor, label
        else:
            return mel_tensor


# ==============================================================================
# Train / Valid 分割
# ==============================================================================
indices = np.arange(len(X_all))
train_indices, valid_indices = train_test_split(
    indices, test_size=TEST_SIZE, random_state=SEED, stratify=y_all
)

X_train = X_all[train_indices]
X_valid = X_all[valid_indices]
y_train = y_all[train_indices]
y_valid = y_all[valid_indices]
train_files_split = [train_files[i] for i in train_indices]
valid_files_split = [train_files[i] for i in valid_indices]

print(f"\nX_train: {X_train.shape}, X_valid: {X_valid.shape}")

# ボーカル積分とジャンル別「ボーカル抜き」適用確率（5:5 になるように）
vocal_integrals_train = None
genre_drop_prob = None
if getattr(config, 'AUGMENT_VOCAL_DROP', False):
    from collections import defaultdict
    th = getattr(config, 'VOCAL_INTEGRAL_THRESHOLD', 2500)
    target_ratio = getattr(config, 'VOCAL_DROP_TARGET_RATIO', 0.5)
    vocal_integrals_train = []
    for f in tqdm(train_files_split, desc="  vocal integral (train)"):
        basename = os.path.splitext(os.path.basename(f))[0]
        cache_path = os.path.join(SEPARATION_CACHE_DIR, f"{basename}.npz")
        data = np.load(cache_path)
        vocal_integrals_train.append(float(np.abs(data['vocals']).sum()))
    by_label = defaultdict(list)
    for i in range(len(y_train)):
        by_label[y_train[i]].append(vocal_integrals_train[i])
    genre_drop_prob = {}
    id_to_name = label_master.set_index('label_id')['label_name'].to_dict()
    print(f"\n  ボーカル抜きオーギュメンテーション内訳 (閾値={th}, 目標割合={target_ratio:.0%})")
    print(f"  {'ジャンル':<12} {'低(≤th)':>8} {'高(>th)':>8} {'適用確率':>10} {'現在の割合':>10} {'期待割合':>10}")
    print("  " + "-" * 58)
    for label_id in range(NUM_CLASSES):
        vals = by_label.get(label_id, [])
        L = sum(1 for v in vals if v <= th)
        H = len(vals) - L
        n = L + H
        p = (target_ratio * (H - L) / H) if H > 0 else 0.0
        genre_drop_prob[label_id] = float(np.clip(p, 0.0, 1.0))
        curr_ratio = L / n if n else 0
        # 適用後の期待: 低 = L + p*H, 期待割合 = (L + p*H) / n
        exp_low = L + genre_drop_prob[label_id] * H
        exp_ratio = exp_low / n if n else 0
        name = id_to_name.get(label_id, f"id{label_id}")
        print(f"  {name:<12} {L:>8} {H:>8} {genre_drop_prob[label_id]:>10.2%} {curr_ratio:>10.1%} {exp_ratio:>10.1%}")
    print("  " + "-" * 58)

# 正規化パラメータは学習データから計算して共有
train_dataset = MelSpectrogramDataset(
    X_train, file_list=train_files_split, y=y_train,
    augment=True, max_frames=max_frames_train,
    vocal_integrals=vocal_integrals_train, genre_drop_prob=genre_drop_prob
)
valid_dataset = MelSpectrogramDataset(
    X_valid, file_list=valid_files_split, y=y_valid,
    augment=False, max_frames=max_frames_train
)
valid_dataset.ch_means = train_dataset.ch_means
valid_dataset.ch_stds = train_dataset.ch_stds

test_dataset = MelSpectrogramDataset(
    X_test_raw, file_list=test_files, y=None,
    augment=False, max_frames=max_frames_train
)
test_dataset.ch_means = train_dataset.ch_means
test_dataset.ch_stds = train_dataset.ch_stds

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=True if torch.cuda.is_available() else False
)
valid_loader = DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=True if torch.cuda.is_available() else False
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=True if torch.cuda.is_available() else False
)

# ==============================================================================
# モデル定義: EfficientNet B3
# ==============================================================================
def build_model(num_classes=NUM_CLASSES):
    """事前学習済み EfficientNet B3 を Fine-tuning 用に構築

    3チャンネル入力:
        ch0 = 原音メルスペクトログラム
        ch1 = 伴奏メルスペクトログラム
        ch2 = ボーカルメルスペクトログラム
    """
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)

    # 前半の層をフリーズ（過学習防止）
    for param in model.features[:6].parameters():
        param.requires_grad = False

    # 分類ヘッドを置き換え
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )

    return model


model = build_model().to(DEVICE)
print(f"\nモデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}")
print(f"学習可能パラメータ数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ==============================================================================
# 損失関数・最適化・スケジューラ
# ==============================================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=WEIGHT_DECAY
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# ==============================================================================
# 学習・評価ループ
# ==============================================================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch_idx, (X_batch, y_batch) in enumerate(loader):
        X_batch, y_batch = X_batch.to(DEVICE, non_blocking=True), y_batch.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * len(y_batch)
        preds = outputs.detach().argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.cpu().numpy())

        del outputs, loss, X_batch, y_batch

        if (batch_idx + 1) % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE, non_blocking=True), y_batch.to(DEVICE, non_blocking=True)

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        total_loss += loss.item() * len(y_batch)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.cpu().numpy())

        del outputs, loss, X_batch, y_batch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


# ==============================================================================
# チェックポイント管理
# ==============================================================================
def save_checkpoint(epoch, model, optimizer, scheduler, best_valid_acc, best_epoch, patience_counter):
    """チェックポイントを保存"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_valid_acc': best_valid_acc,
        'best_epoch': best_epoch,
        'patience_counter': patience_counter,
    }
    torch.save(checkpoint, CHECKPOINT_FILE)
    print(f"  → チェックポイント保存: {CHECKPOINT_FILE}")


def load_checkpoint(model, optimizer, scheduler, load_optimizer=True, load_scheduler=True):
    """チェックポイントから学習を再開"""
    if os.path.exists(CHECKPOINT_FILE):
        print(f"チェックポイントを読み込み: {CHECKPOINT_FILE}")
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)

        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  → モデルの重みを読み込みました")

        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"  → 最適化器の状態を読み込みました")
            except Exception as e:
                print(f"  ⚠ 最適化器の状態の読み込みに失敗: {e}")
                print(f"  → 最適化器は初期状態から開始します")
        else:
            print(f"  → 最適化器は初期状態から開始します（設定によりスキップ）")

        if load_scheduler and 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"  → スケジューラの状態を読み込みました")
            except Exception as e:
                print(f"  ⚠ スケジューラの状態の読み込みに失敗: {e}")
                print(f"  → スケジューラは初期状態から開始します")
        else:
            print(f"  → スケジューラは初期状態から開始します（設定によりスキップ）")

        start_epoch = checkpoint.get('epoch', 0) + 1
        best_valid_acc = checkpoint.get('best_valid_acc', 0)
        best_epoch = checkpoint.get('best_epoch', 0)
        patience_counter = checkpoint.get('patience_counter', 0)

        print(f"  → 再開: epoch {start_epoch} から")
        print(f"  → ベスト精度: {best_valid_acc:.4f} (epoch {best_epoch})")
        return start_epoch, best_valid_acc, best_epoch, patience_counter
    return 1, 0, 0, 0


# シグナルハンドラ（Ctrl+Cで安全に停止）
interrupted = False
def signal_handler(sig, frame):
    global interrupted
    print("\n\n中断シグナルを受信しました。チェックポイントを保存して終了します...")
    interrupted = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ==============================================================================
# 学習実行
# ==============================================================================
print("\n=== 学習開始 ===")

if RESUME_FROM_CHECKPOINT:
    start_epoch, best_valid_acc, best_epoch, patience_counter = load_checkpoint(
        model, optimizer, scheduler,
        load_optimizer=LOAD_OPTIMIZER_STATE,
        load_scheduler=LOAD_SCHEDULER_STATE
    )
else:
    print("チェックポイントの読み込みをスキップします（最初から学習）")
    start_epoch, best_valid_acc, best_epoch, patience_counter = 1, 0, 0, 0

patience = PATIENCE

current_epoch = start_epoch

print(f"学習設定:")
print(f"  開始エポック: {start_epoch}")
print(f"  最大エポック: {EPOCHS}")
print(f"  バッチサイズ: {BATCH_SIZE}")
print(f"  学習データ数: {len(train_dataset)}")
print(f"  検証データ数: {len(valid_dataset)}")
print(f"  入力チャンネル: 3ch (原音 / 伴奏 / ボーカル)")
print(f"  デバイス: {DEVICE}")

try:
    for epoch in range(start_epoch, EPOCHS + 1):
        current_epoch = epoch

        if interrupted:
            print("\n学習を中断します...")
            save_checkpoint(epoch - 1, model, optimizer, scheduler, best_valid_acc, best_epoch, patience_counter)
            break

        print(f"Epoch {epoch} の学習を開始...")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch} の評価を開始...")
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Valid Loss: {valid_loss:.4f} Acc: {valid_acc:.4f} | "
              f"LR: {lr:.6f}")

        if valid_acc >= best_valid_acc:
            best_valid_acc = valid_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_FILE)
            save_checkpoint(epoch, model, optimizer, scheduler, best_valid_acc, best_epoch, patience_counter)
            print(f"  → ベストモデル更新 (Valid Acc: {best_valid_acc:.4f})")
        else:
            patience_counter += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if patience_counter >= patience:
            print(f"\nEarly Stopping at epoch {epoch} (ベスト: epoch {best_epoch})")
            break

    print(f"\n=== 学習完了 ===")
    print(f"ベストモデル: epoch {best_epoch}, Valid Acc: {best_valid_acc:.4f}")

except KeyboardInterrupt:
    print("\n\nCtrl+Cで中断されました。チェックポイントを保存します...")
    save_checkpoint(current_epoch - 1, model, optimizer, scheduler, best_valid_acc, best_epoch, patience_counter)
    print("チェックポイント保存完了。次回は同じスクリプトで再開できます。")
except Exception as e:
    print(f"\n\nエラーが発生しました: {type(e).__name__}")
    print(f"エラーメッセージ: {str(e)}")
    import traceback
    traceback.print_exc()
    print("\nチェックポイントを保存します...")
    if current_epoch > start_epoch:
        save_checkpoint(current_epoch - 1, model, optimizer, scheduler, best_valid_acc, best_epoch, patience_counter)
    sys.exit(1)

finally:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPUメモリをクリアしました。")

# ==============================================================================
# テストデータの推論・提出ファイル作成
# ==============================================================================
print("\n=== テストデータ推論 ===")
if os.path.exists(BEST_MODEL_FILE):
    model.load_state_dict(torch.load(BEST_MODEL_FILE, map_location=DEVICE))
    print(f"ベストモデルを読み込み: {BEST_MODEL_FILE}")
else:
    print(f"警告: {BEST_MODEL_FILE} が見つかりません。チェックポイントから読み込みます。")
    if os.path.exists(CHECKPOINT_FILE):
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"チェックポイントから読み込み: {CHECKPOINT_FILE}")
    else:
        print("エラー: モデルファイルが見つかりません。")
        sys.exit(1)
model.eval()

all_preds = []
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(DEVICE, non_blocking=True)
        outputs = model(X_batch)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)

        del outputs, X_batch

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 提出CSV作成
test_file_names = [os.path.basename(f) for f in test_files]
submit_df = pd.DataFrame({
    0: test_file_names,
    1: all_preds
})
submit_df.to_csv(SUBMIT_FILE, index=False, header=False)
print(f"提出ファイル作成完了: {SUBMIT_FILE} ({len(submit_df)} 件)")
print(submit_df.head(10))
