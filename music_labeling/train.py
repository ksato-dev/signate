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

from sklearn.model_selection import StratifiedKFold
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
N_FOLDS = config.N_FOLDS
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
    n_fft = 2048
    hop_length = 512
    kwargs = {} if n_mels is None else {'n_mels': n_mels}
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, **kwargs)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db



# ==============================================================================
# オーギュメンテーション（7パターン方式）
# 元音声 + 6種の拡張を事前に定義し、毎回ランダムに1つ選択
# ==============================================================================
AUGMENT_TYPES = [
    'original',       # 元音声そのまま
    'pitch_up',       # ピッチを上げる
    'pitch_down',     # ピッチを下げる
    'stretch_slow',   # テンポを遅くする
    'stretch_fast',   # テンポを速くする
    'noise',          # ガウシアンノイズ追加
    'time_shift',     # 循環シフト
]


def generate_augment_variant(sr=SR):
    """7パターンからランダムに1つ選び、パラメータを生成

    3チャンネル（原音・伴奏・ボーカル）に同一パラメータを適用するため、
    パラメータを事前に決定して返す。
    """
    aug_type = np.random.choice(AUGMENT_TYPES)
    params = {'type': aug_type}

    if aug_type == 'pitch_up':
        params['n_steps'] = np.random.uniform(config.PITCH_UP_MIN, config.PITCH_UP_MAX)
    elif aug_type == 'pitch_down':
        params['n_steps'] = np.random.uniform(config.PITCH_DOWN_MIN, config.PITCH_DOWN_MAX)
    elif aug_type == 'stretch_slow':
        params['rate'] = np.random.uniform(config.STRETCH_SLOW_MIN, config.STRETCH_SLOW_MAX)
    elif aug_type == 'stretch_fast':
        params['rate'] = np.random.uniform(config.STRETCH_FAST_MIN, config.STRETCH_FAST_MAX)
    elif aug_type == 'noise':
        params['noise_level'] = np.random.uniform(config.NOISE_LEVEL_MIN, config.NOISE_LEVEL_MAX)
    elif aug_type == 'time_shift':
        params['roll_frac'] = np.random.uniform(config.ROLL_FRAC_MIN, config.ROLL_FRAC_MAX)
        params['roll_sign'] = np.random.choice([-1, 1])

    return params


def apply_augment_variant(y, params, sr=SR, noise_seed=None):
    """生成済みパラメータで1種類のオーギュメンテーションを適用し、長さを統一

    Args:
        y: 音声波形
        params: generate_augment_variant() で生成したパラメータ
        sr: サンプリングレート
        noise_seed: ノイズ用シード（3チャンネルに同じノイズを適用する場合）
    """
    aug_type = params['type']

    if aug_type in ('pitch_up', 'pitch_down'):
        y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=params['n_steps'])
    elif aug_type in ('stretch_slow', 'stretch_fast'):
        y = librosa.effects.time_stretch(y=y, rate=params['rate'])
    elif aug_type == 'noise':
        if noise_seed is not None:
            rng = np.random.RandomState(noise_seed)
            noise = rng.normal(0, params['noise_level'], y.shape)
        else:
            noise = np.random.normal(0, params['noise_level'], y.shape)
        y = y + noise
    elif aug_type == 'time_shift':
        shift = int(len(y) * params['roll_frac']) * params['roll_sign']
        y = np.roll(y, shift)

    target_len = int(sr * getattr(config, 'AUDIO_DURATION', 30))
    y = librosa.util.fix_length(y, size=target_len)

    return y


# ==============================================================================
# 画像系オーギュメンテーション（CutMix / MixUp / CutOut）
# バッチ単位で適用。学習ループ内で使用。
# ==============================================================================
def cutmix(x, y, alpha=1.0):
    """CutMix: 2サンプルを矩形領域で切り貼りし、ラベルは面積比で混合

    Args:
        x: (B, C, H, W) 入力バッチ
        y: (B,) ラベル
        alpha: Beta(alpha, alpha) から lam をサンプル（lam が切り取る面積比の目安）

    Returns:
        mixed_x: (B, C, H, W), y_a: (B,), y_b: (B,), lam: (B,) スカラーは 1-lam が貼り付け面積比
    """
    B, C, H, W = x.shape
    lam = np.random.beta(alpha, alpha)
    # 貼り付け矩形のサイズ（面積がおおよそ 1-lam になるように）
    cut_rat = np.sqrt(1.0 - lam)
    w_cut = int(W * cut_rat)
    h_cut = int(H * cut_rat)
    # ランダムな中心
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - w_cut // 2, 0, W)
    x2 = np.clip(cx + w_cut // 2, 0, W)
    y1 = np.clip(cy - h_cut // 2, 0, H)
    y2 = np.clip(cy + h_cut // 2, 0, H)
    # 実際の面積比で lam を再計算（損失の重み用）
    area = (x2 - x1) * (y2 - y1)
    lam = 1.0 - (area / (H * W))

    # シャッフル: 各サンプル i に、別のサンプル perm[i] の矩形を貼る
    perm = np.random.permutation(B)
    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
    y_a, y_b = y, y[perm]
    return mixed_x, y_a, y_b, lam


def mixup(x, y, alpha=0.2):
    """MixUp: 2サンプルをピクセル単位で線形混合

    Args:
        x: (B, C, H, W)
        y: (B,)
        alpha: Beta(alpha, alpha)

    Returns:
        mixed_x, y_a, y_b, lam (スカラー or per-sample)
    """
    B = x.shape[0]
    lam = np.random.beta(alpha, alpha)
    perm = np.random.permutation(B)
    mixed_x = lam * x + (1.0 - lam) * x[perm]
    y_a, y_b = y, y[perm]
    return mixed_x, y_a, y_b, lam


def cutout(x, n_holes=1, ratio=0.15):
    """CutOut: 画像内の矩形領域をゼロでマスク（チャンネル共通）

    Args:
        x: (B, C, H, W)
        n_holes: マスクする矩形の数
        ratio: 辺の長さの画像に対する比率 (0〜1)
    """
    B, C, H, W = x.shape
    length = int(min(H, W) * ratio)
    if length <= 0:
        return x
    out = x.clone()
    for _ in range(n_holes):
        cy = np.random.randint(H)
        cx = np.random.randint(W)
        y1 = np.clip(cy - length // 2, 0, H)
        y2 = np.clip(cy + length // 2, 0, H)
        x1 = np.clip(cx - length // 2, 0, W)
        x2 = np.clip(cx + length // 2, 0, W)
        out[:, :, y1:y2, x1:x2] = 0
    return out


def _apply_mix_augment(x, y):
    """CutMix / MixUp のいずれかを確率で適用。適用時は (mixed_x, y_a, y_b, lam)、しなければ (x, None, None, None)。"""
    use_cutmix = getattr(config, 'AUGMENT_CUTMIX', False) and np.random.random() < getattr(config, 'AUGMENT_CUTMIX_PROB', 0.5)
    use_mixup = getattr(config, 'AUGMENT_MIXUP', False) and np.random.random() < getattr(config, 'AUGMENT_MIXUP_PROB', 0.5)
    if use_cutmix:
        alpha = getattr(config, 'AUGMENT_CUTMIX_ALPHA', 1.0)
        return cutmix(x, y, alpha=alpha)
    if use_mixup:
        alpha = getattr(config, 'AUGMENT_MIXUP_ALPHA', 0.2)
        return mixup(x, y, alpha=alpha)
    return x, None, None, None


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


def load_mel_cache_if_valid(train_files, test_files, cache_path=None):
    """事前計算済みメルスペクトログラムキャッシュがあればロードする。
    ファイルリストが一致する場合のみ有効。不一致なら None を返す。
    Returns:
        (X_all, X_test_raw, max_frames_train) or None if cache miss
    """
    cache_path = cache_path or getattr(config, 'MEL_CACHE_FILE', 'data/precomputed_mel.npz')
    if not os.path.isfile(cache_path):
        return None
    train_basenames = [os.path.splitext(os.path.basename(f))[0] for f in train_files]
    test_basenames = [os.path.splitext(os.path.basename(f))[0] for f in test_files]
    try:
        data = np.load(cache_path, allow_pickle=True)
        cached_train = list(data.get('train_basenames', []))
        cached_test = list(data.get('test_basenames', []))
        if cached_train != train_basenames or cached_test != test_basenames:
            return None
        X_all = data['X_all']
        X_test_raw = data['X_test_raw']
        max_frames_train = int(data['max_frames_train'])
        return X_all, X_test_raw, max_frames_train
    except Exception:
        return None


def save_mel_cache(X_all, X_test_raw, max_frames_train, train_files, test_files, cache_path=None):
    """事前計算済みメルスペクトログラムをキャッシュに保存する。"""
    cache_path = cache_path or getattr(config, 'MEL_CACHE_FILE', 'data/precomputed_mel.npz')
    train_basenames = [os.path.splitext(os.path.basename(f))[0] for f in train_files]
    test_basenames = [os.path.splitext(os.path.basename(f))[0] for f in test_files]
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        X_all=X_all,
        X_test_raw=X_test_raw,
        max_frames_train=np.int64(max_frames_train),
        train_basenames=np.array(train_basenames, dtype=object),
        test_basenames=np.array(test_basenames, dtype=object),
    )


# ==============================================================================
# MagnaTagATune 外部データセット読み込み
# ==============================================================================
MAGNA_GENRE_COL_MAP = {
    'blues': 'blues',
    'classical': 'classical',
    'country': 'country',
    'disco': 'disco',
    'hiphop': 'hip hop',
    'jazz': 'jazz',
    'metal': 'metal',
    'pop': 'pop',
    'reggae': 'reggae',
    'rock': 'rock',
}


def load_magna_data(label_master, dataset_dir, annotations_file,
                    samples_per_class, seed=SEED):
    """MagnaTagATune からジャンルラベル付きサンプルを均等サンプリングして返す

    1. annotations CSV の 10 ジャンルカラムを label_master と照合
    2. 複数ジャンルタグを持つクリップは最も件数が少ないジャンルに割り当て
    3. クラスごとに samples_per_class 件を均等サンプリング

    Returns:
        (file_list, labels): ファイルパスのリストと label_id の numpy 配列
    """
    ann = pd.read_csv(annotations_file, sep='\t')
    label_name_to_id = label_master.set_index('label_name')['label_id'].to_dict()

    genre_columns = list(MAGNA_GENRE_COL_MAP.values())
    genre_matrix = ann[genre_columns].values  # (n_clips, 10)
    n_match = genre_matrix.sum(axis=1)

    # mp3_path → 実ファイルパスのマッピングを構築
    numbered_dirs = sorted(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and d.isdigit()
    )
    path_map = {}
    for nd in numbered_dirs:
        nd_path = os.path.join(dataset_dir, nd)
        for hex_dir in os.listdir(nd_path):
            hex_path = os.path.join(nd_path, hex_dir)
            if not os.path.isdir(hex_path):
                continue
            for fname in os.listdir(hex_path):
                if fname.endswith('.mp3'):
                    path_map[f'{hex_dir}/{fname}'] = os.path.join(hex_path, fname)

    # 各ジャンルのグローバル件数（優先度決定用: 少ないジャンルを優先）
    genre_totals = genre_matrix.sum(axis=0)
    inv_map = {v: k for k, v in MAGNA_GENRE_COL_MAP.items()}

    by_class = {lid: [] for lid in range(len(label_name_to_id))}
    for i in range(len(ann)):
        if n_match[i] == 0:
            continue
        mp3_path = ann.iloc[i]['mp3_path']
        if mp3_path not in path_map:
            continue
        matched = [j for j in range(len(genre_columns)) if genre_matrix[i, j] == 1]
        best = min(matched, key=lambda j: genre_totals[j])
        genre_name = inv_map[genre_columns[best]]
        label_id = label_name_to_id[genre_name]
        by_class[label_id].append(path_map[mp3_path])

    # クラスごとに均等サンプリング
    rng = np.random.RandomState(seed)
    sampled_files, sampled_labels = [], []
    id_to_name = {v: k for k, v in label_name_to_id.items()}

    print(f"  MagnaTagATune ジャンル別利用可能数:")
    for label_id in range(len(label_name_to_id)):
        candidates = by_class[label_id]
        n = min(samples_per_class, len(candidates))
        name = id_to_name.get(label_id, f'id{label_id}')
        print(f"    {name:<12} {len(candidates):>6} 件 → {n} 件サンプリング")
        if n > 0:
            chosen = rng.choice(candidates, n, replace=False)
            sampled_files.extend(chosen)
            sampled_labels.extend([label_id] * n)

    return sampled_files, np.array(sampled_labels, dtype=np.int64)


def load_gtzan_data(label_master, genres_dir):
    """GTZAN Dataset からジャンルフォルダ構造を使って全音声ファイルを読み込む

    フォルダ名が label_master の label_name と完全一致する前提。

    Returns:
        (file_list, labels): ファイルパスのリストと label_id の numpy 配列
    """
    label_name_to_id = label_master.set_index('label_name')['label_id'].to_dict()
    files, labels = [], []

    for genre_name in sorted(os.listdir(genres_dir)):
        genre_path = os.path.join(genres_dir, genre_name)
        if not os.path.isdir(genre_path):
            continue
        if genre_name not in label_name_to_id:
            print(f"  [SKIP] GTZAN フォルダ '{genre_name}' は label_master に存在しません")
            continue
        label_id = label_name_to_id[genre_name]
        genre_files = natsorted(
            os.path.join(genre_path, f)
            for f in os.listdir(genre_path)
            if f.endswith(('.wav', '.au', '.mp3'))
        )
        files.extend(genre_files)
        labels.extend([label_id] * len(genre_files))
        print(f"    {genre_name:<12} {len(genre_files):>4} 件")

    return files, np.array(labels, dtype=np.int64)


# ==============================================================================
# 音源分離の実行（Windows multiprocessing のため __main__ 内で実行）
# ==============================================================================
if __name__ == '__main__':
    print("=== データファイルの読み込み ===")
    train_master = pd.read_csv('data/train_master.csv', index_col=0)
    label_master = pd.read_csv('data/label_master.csv')

    train_files = natsorted(glob.glob('data/train_sound_*/train_sound_*/train_*.au'))
    test_files = natsorted(glob.glob('data/test_sound_*/test_sound_*/test_*.au'))
    n_original_train = len(train_files)
    print(f"学習ファイル数: {n_original_train}")
    print(f"テストファイル数: {len(test_files)}")

    # MagnaTagATune 外部データの追加
    magna_labels = None
    if getattr(config, 'USE_MAGNA_DATA', False):
        magna_dir = getattr(config, 'MAGNA_DATASET_DIR', 'data/TheMagnaTagATuneDataset')
        magna_ann = getattr(config, 'MAGNA_ANNOTATIONS_FILE',
                            'data/TheMagnaTagATuneDataset/annotations_final.csv')
        magna_n = getattr(config, 'MAGNA_SAMPLES_PER_CLASS', 50)
        print(f"\n=== MagnaTagATune 外部データ読み込み ({magna_n} 件/クラス) ===")
        magna_files, magna_labels = load_magna_data(
            label_master, magna_dir, magna_ann, magna_n, seed=SEED)
        train_files = train_files + magna_files
        print(f"  追加: {len(magna_files)} 件 → 学習ファイル合計: {len(train_files)} 件")

        id_to_name = label_master.set_index('label_id')['label_name'].to_dict()
        magna_profile = pd.DataFrame({
            'file_path': magna_files,
            'label_id': magna_labels,
            'label_name': [id_to_name[lid] for lid in magna_labels],
        })
        magna_csv_path = 'data/magna_sampled.csv'
        magna_profile.to_csv(magna_csv_path, index=False)
        print(f"  サンプリング結果を保存: {magna_csv_path} ({len(magna_profile)} 件)")

    # GTZAN 外部データの追加
    gtzan_labels = None
    if getattr(config, 'USE_GTZAN_DATA', False):
        gtzan_dir = getattr(config, 'GTZAN_GENRES_DIR',
                            'data/GTZAN_Dataset/Data/genres_original')
        print(f"\n=== GTZAN 外部データ読み込み ===")
        gtzan_files, gtzan_labels = load_gtzan_data(label_master, gtzan_dir)
        train_files = train_files + gtzan_files
        print(f"  追加: {len(gtzan_files)} 件 → 学習ファイル合計: {len(train_files)} 件")

        id_to_name = label_master.set_index('label_id')['label_name'].to_dict()
        gtzan_profile = pd.DataFrame({
            'file_path': gtzan_files,
            'label_id': gtzan_labels,
            'label_name': [id_to_name[lid] for lid in gtzan_labels],
        })
        gtzan_csv_path = 'data/gtzan_profile.csv'
        gtzan_profile.to_csv(gtzan_csv_path, index=False)
        print(f"  プロファイルを保存: {gtzan_csv_path} ({len(gtzan_profile)} 件)")

    # オーギュメンテーション設定プロファイルを CSV に出力
    aug_profile = {
        'method': '7-variant random selection',
        'audio_duration_sec': config.AUDIO_DURATION,
        'pitch_up_min': config.PITCH_UP_MIN,
        'pitch_up_max': config.PITCH_UP_MAX,
        'pitch_down_min': config.PITCH_DOWN_MIN,
        'pitch_down_max': config.PITCH_DOWN_MAX,
        'stretch_slow_min': config.STRETCH_SLOW_MIN,
        'stretch_slow_max': config.STRETCH_SLOW_MAX,
        'stretch_fast_min': config.STRETCH_FAST_MIN,
        'stretch_fast_max': config.STRETCH_FAST_MAX,
        'noise_level_min': config.NOISE_LEVEL_MIN,
        'noise_level_max': config.NOISE_LEVEL_MAX,
        'roll_frac_min': config.ROLL_FRAC_MIN,
        'roll_frac_max': config.ROLL_FRAC_MAX,
        'time_masking': config.AUGMENT_TIME_MASKING,
        'frequency_masking': config.AUGMENT_FREQUENCY_MASKING,
        'cutmix': getattr(config, 'AUGMENT_CUTMIX', False),
        'cutmix_prob': getattr(config, 'AUGMENT_CUTMIX_PROB', 0),
        'cutmix_alpha': getattr(config, 'AUGMENT_CUTMIX_ALPHA', 0),
        'mixup': getattr(config, 'AUGMENT_MIXUP', False),
        'mixup_prob': getattr(config, 'AUGMENT_MIXUP_PROB', 0),
        'mixup_alpha': getattr(config, 'AUGMENT_MIXUP_ALPHA', 0),
        'cutout': getattr(config, 'AUGMENT_CUTOUT', False),
        'cutout_prob': getattr(config, 'AUGMENT_CUTOUT_PROB', 0),
        'cutout_n_holes': getattr(config, 'AUGMENT_CUTOUT_N_HOLES', 0),
        'cutout_ratio': getattr(config, 'AUGMENT_CUTOUT_RATIO', 0),
    }
    aug_df = pd.DataFrame([aug_profile])
    aug_csv_path = 'data/augmentation_profile.csv'
    os.makedirs(os.path.dirname(aug_csv_path), exist_ok=True)
    aug_df.to_csv(aug_csv_path, index=False)
    print(f"\nオーギュメンテーション設定を保存: {aug_csv_path}")

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
    # 3チャンネル メルスペクトログラム（キャッシュがあればロード短縮）
    # ==============================================================================
    mel_cache_path = getattr(config, 'MEL_CACHE_FILE', 'data/precomputed_mel.npz')
    cached = load_mel_cache_if_valid(train_files, test_files, mel_cache_path)
    if cached is not None:
        X_all, X_test_raw, max_frames_train = cached
        print("\n=== 3チャンネル メルスペクトログラム（事前キャッシュからロード） ===")
        print(f"X_all shape: {X_all.shape}, X_test shape: {X_test_raw.shape}, max_frames: {max_frames_train}")
    else:
        print("\n=== 3チャンネル メルスペクトログラム抽出 ===")
        mel_freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=0.0, fmax=SR / 2.0)
        print(f"メル周波数バンド: {N_MELS}バンド, {mel_freqs[0]:.1f} - {mel_freqs[-1]:.1f} Hz")

        print("\n学習データ:")
        X_all, max_frames_train = load_and_pad_3ch(train_files, SEPARATION_CACHE_DIR, SR, N_MELS)
        print(f"X_all shape: {X_all.shape}")

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
        save_mel_cache(X_all, X_test_raw, max_frames_train, train_files, test_files, mel_cache_path)
        print(f"  メルキャッシュを保存しました: {mel_cache_path}")

    y_all = train_master['label_id'].values
    n_magna = 0
    if magna_labels is not None and len(magna_labels) > 0:
        y_all = np.concatenate([y_all, magna_labels])
        n_magna = len(magna_labels)
    n_gtzan = 0
    if gtzan_labels is not None and len(gtzan_labels) > 0:
        y_all = np.concatenate([y_all, gtzan_labels])
        n_gtzan = len(gtzan_labels)
    print(f"y_all shape: {y_all.shape} (original: {n_original_train}, magna: {n_magna}, gtzan: {n_gtzan})")


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
                 cache_dir=SEPARATION_CACHE_DIR):
        self.X = X  # (N, 3, n_mels, time_frames)
        self.file_list = file_list
        self.y = y
        self.img_size = img_size
        self.augment = augment
        self.sr = sr
        self.n_mels = n_mels
        self.max_frames = max_frames
        self.cache_dir = cache_dir

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

            # 7パターンからランダムに1つ選択し、3チャンネルに同一適用
            aug_params = generate_augment_variant(sr=self.sr)
            noise_seed = np.random.randint(0, 2**31)

            original = apply_augment_variant(original, aug_params,
                                             sr=self.sr, noise_seed=noise_seed)
            accompaniment = apply_augment_variant(accompaniment, aug_params,
                                                  sr=self.sr, noise_seed=noise_seed)
            vocals = apply_augment_variant(vocals, aug_params,
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
# K-Fold 学習ヘルパー
# ==============================================================================
def run_training_loop(model, train_loader, valid_loader, criterion, optimizer,
                      scheduler, fold=0, start_epoch=1, best_valid_acc=0,
                      best_epoch=0, patience_counter=0):
    """1 Fold / 全データ学習の共通学習ループ

    valid_loader が None の場合は Early Stopping を行わず EPOCHS まで学習する。
    Returns:
        best_valid_acc, best_epoch
    """
    best_model_path = BEST_MODEL_FILE.format(fold)
    current_epoch = start_epoch

    try:
        for epoch in range(start_epoch, EPOCHS + 1):
            current_epoch = epoch

            if interrupted:
                print("\n学習を中断します...")
                save_checkpoint(epoch - 1, model, optimizer, scheduler,
                                best_valid_acc, best_epoch, patience_counter, fold=fold)
                break

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)

            if valid_loader is not None:
                valid_loss, valid_acc = evaluate(model, valid_loader, criterion)
            else:
                valid_loss, valid_acc = 0.0, 0.0

            scheduler.step()
            lr = optimizer.param_groups[0]['lr']

            if valid_loader is not None:
                print(f"Epoch {epoch:3d}/{EPOCHS} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                      f"Valid Loss: {valid_loss:.4f} Acc: {valid_acc:.4f} | "
                      f"LR: {lr:.6f}")
            else:
                print(f"Epoch {epoch:3d}/{EPOCHS} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                      f"LR: {lr:.6f}")

            if valid_loader is not None:
                if valid_acc >= best_valid_acc:
                    best_valid_acc = valid_acc
                    best_epoch = epoch
                    patience_counter = 0
                    torch.save(model.state_dict(), best_model_path)
                    save_checkpoint(epoch, model, optimizer, scheduler,
                                    best_valid_acc, best_epoch, patience_counter, fold=fold)
                    print(f"  -> Best model updated (Valid Acc: {best_valid_acc:.4f})")
                else:
                    patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"\nEarly Stopping at epoch {epoch} (best: epoch {best_epoch})")
                    break
            else:
                best_valid_acc = train_acc
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_path)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\n\nCtrl+C: saving checkpoint...")
        save_checkpoint(current_epoch - 1, model, optimizer, scheduler,
                        best_valid_acc, best_epoch, patience_counter, fold=fold)
    except Exception as e:
        print(f"\n\nError: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        if current_epoch > start_epoch:
            save_checkpoint(current_epoch - 1, model, optimizer, scheduler,
                            best_valid_acc, best_epoch, patience_counter, fold=fold)
        sys.exit(1)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return best_valid_acc, best_epoch

# ==============================================================================
# モデル定義: EfficientNet B3（モジュールレベルで定義、ワーカーから利用）
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


if __name__ == '__main__':
    pass  # model / optimizer / scheduler は Fold ループ内で生成

# ==============================================================================
# 学習・評価ループ
# ==============================================================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch_idx, (X_batch, y_batch) in enumerate(loader):
        X_batch = X_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.to(DEVICE, non_blocking=True)

        # CutMix / MixUp（確率で適用）
        mixed_x, y_a, y_b, lam = _apply_mix_augment(X_batch, y_batch)
        if y_a is not None:
            X_batch = mixed_x  # 混合済みで上書き

        # CutOut（確率で適用）
        if getattr(config, 'AUGMENT_CUTOUT', False) and np.random.random() < getattr(config, 'AUGMENT_CUTOUT_PROB', 0.5):
            n_holes = getattr(config, 'AUGMENT_CUTOUT_N_HOLES', 1)
            ratio = getattr(config, 'AUGMENT_CUTOUT_RATIO', 0.15)
            X_batch = cutout(X_batch, n_holes=n_holes, ratio=ratio)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(X_batch)

        if y_a is not None:
            # Mix/CutMix 時: ラベルを面積比で混合した損失
            loss = lam * criterion(outputs, y_a) + (1.0 - lam) * criterion(outputs, y_b)
        else:
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
def save_checkpoint(epoch, model, optimizer, scheduler, best_valid_acc, best_epoch, patience_counter, fold=0):
    """チェックポイントを保存"""
    ckpt_path = CHECKPOINT_FILE.format(fold)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_valid_acc': best_valid_acc,
        'best_epoch': best_epoch,
        'patience_counter': patience_counter,
    }
    torch.save(checkpoint, ckpt_path)
    print(f"  → チェックポイント保存: {ckpt_path}")


def load_checkpoint(model, optimizer, scheduler, fold=0, load_optimizer=True, load_scheduler=True):
    """チェックポイントから学習を再開"""
    ckpt_path = CHECKPOINT_FILE.format(fold)
    if os.path.exists(ckpt_path):
        print(f"チェックポイントを読み込み: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)

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
# K-Fold 交差検証 + 全データ再学習（__main__ 内で実行）
# ==============================================================================
if __name__ == '__main__':
    indices = np.arange(len(X_all))
    fold_scores = []
    FINAL_FOLD = max(N_FOLDS, 1)  # fold 番号: CV 時は N_FOLDS、CV なしは 1

    # ==================================================================
    # K-Fold 交差検証（N_FOLDS >= 2 のとき実行）
    # ==================================================================
    if N_FOLDS >= 2:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        print(f"\n=== {N_FOLDS}-Fold 交差検証 ===")

        for fold, (train_indices, valid_indices) in enumerate(skf.split(indices, y_all)):
            if interrupted:
                break

            print(f"\n{'='*60}")
            print(f"  Fold {fold + 1}/{N_FOLDS}")
            print(f"{'='*60}")

            X_train = X_all[train_indices]
            X_valid = X_all[valid_indices]
            y_train = y_all[train_indices]
            y_valid = y_all[valid_indices]
            train_files_fold = [train_files[i] for i in train_indices]
            valid_files_fold = [train_files[i] for i in valid_indices]

            print(f"  Train: {len(X_train)}, Valid: {len(X_valid)}")

            train_dataset = MelSpectrogramDataset(
                X_train, file_list=train_files_fold, y=y_train,
                augment=True, max_frames=max_frames_train,
            )
            valid_dataset = MelSpectrogramDataset(
                X_valid, file_list=valid_files_fold, y=y_valid,
                augment=False, max_frames=max_frames_train,
            )
            valid_dataset.ch_means = train_dataset.ch_means
            valid_dataset.ch_stds = train_dataset.ch_stds

            train_loader = DataLoader(
                train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                num_workers=12, persistent_workers=True,
                pin_memory=torch.cuda.is_available(),
            )
            valid_loader = DataLoader(
                valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=12, persistent_workers=True,
                pin_memory=torch.cuda.is_available(),
            )

            model = build_model().to(DEVICE)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LR, weight_decay=WEIGHT_DECAY,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

            if RESUME_FROM_CHECKPOINT:
                start_epoch, best_valid_acc, best_epoch, patience_counter = load_checkpoint(
                    model, optimizer, scheduler, fold=fold,
                    load_optimizer=LOAD_OPTIMIZER_STATE,
                    load_scheduler=LOAD_SCHEDULER_STATE,
                )
            else:
                start_epoch, best_valid_acc, best_epoch, patience_counter = 1, 0, 0, 0

            best_valid_acc, best_epoch = run_training_loop(
                model, train_loader, valid_loader, criterion, optimizer, scheduler,
                fold=fold, start_epoch=start_epoch, best_valid_acc=best_valid_acc,
                best_epoch=best_epoch, patience_counter=patience_counter,
            )

            fold_scores.append(best_valid_acc)
            print(f"\n  Fold {fold + 1} best: epoch {best_epoch}, Valid Acc: {best_valid_acc:.4f}")

            del model, optimizer, scheduler, train_loader, valid_loader
            del train_dataset, valid_dataset
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # CV スコアサマリー
        if fold_scores:
            print(f"\n{'='*60}")
            print(f"  CV Summary ({N_FOLDS}-Fold)")
            print(f"{'='*60}")
            for i, score in enumerate(fold_scores):
                print(f"  Fold {i + 1}: {score:.4f}")
            print(f"  Mean: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")
            print(f"{'='*60}")

        # CV 後の待機（Ctrl+C で再学習をスキップ可能）
        if not interrupted and fold_scores:
            import time
            wait_sec = 20
            print(f"\n全データで再学習を開始します。中止するには Ctrl+C を押してください ({wait_sec}秒待機)...")
            try:
                time.sleep(wait_sec)
            except KeyboardInterrupt:
                interrupted = True
                print("\n再学習をスキップします。")

    # ==================================================================
    # 全データで最終モデルを学習
    # ==================================================================
    if not interrupted:
        print(f"\n{'='*60}")
        print(f"  Final training on ALL data")
        print(f"{'='*60}")

        full_dataset = MelSpectrogramDataset(
            X_all, file_list=train_files, y=y_all,
            augment=True, max_frames=max_frames_train,
        )
        full_loader = DataLoader(
            full_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=12, persistent_workers=True,
            pin_memory=torch.cuda.is_available(),
        )

        # CV 実施済みなら平均ベストエポック数で学習、そうでなければ EPOCHS まで
        if fold_scores:
            cv_best_epochs = []
            for fi in range(N_FOLDS):
                ckpt_path = CHECKPOINT_FILE.format(fi)
                if os.path.exists(ckpt_path):
                    ckpt = torch.load(ckpt_path, map_location='cpu')
                    cv_best_epochs.append(ckpt.get('best_epoch', EPOCHS))
            final_epochs = max(1, int(np.mean(cv_best_epochs))) if cv_best_epochs else EPOCHS
        else:
            final_epochs = EPOCHS

        saved_epochs = EPOCHS
        EPOCHS = final_epochs
        print(f"  Epochs: {EPOCHS}" + (f" (CV average best epoch)" if fold_scores else ""))

        model = build_model().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR, weight_decay=WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=saved_epochs, eta_min=1e-6)

        if RESUME_FROM_CHECKPOINT:
            start_epoch, _, _, _ = load_checkpoint(
                model, optimizer, scheduler, fold=FINAL_FOLD,
                load_optimizer=LOAD_OPTIMIZER_STATE,
                load_scheduler=LOAD_SCHEDULER_STATE,
            )
        else:
            start_epoch = 1

        run_training_loop(
            model, full_loader, None, criterion, optimizer, scheduler,
            fold=FINAL_FOLD, start_epoch=start_epoch,
        )

        EPOCHS = saved_epochs

        del full_loader, full_dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ==================================================================
    # テストデータの推論・提出ファイル作成
    # ==================================================================
    print("\n=== テストデータ推論 ===")

    final_model_path = BEST_MODEL_FILE.format(FINAL_FOLD)
    if os.path.exists(final_model_path):
        model = build_model().to(DEVICE)
        model.load_state_dict(torch.load(final_model_path, map_location=DEVICE))
        print(f"最終モデルを読み込み: {final_model_path}")
    else:
        print(f"エラー: {final_model_path} が見つかりません。")
        sys.exit(1)

    # テスト用 Dataset は全学習データの正規化パラメータを使用
    norm_dataset = MelSpectrogramDataset(
        X_all, file_list=train_files, y=y_all,
        augment=False, max_frames=max_frames_train,
    )
    test_dataset = MelSpectrogramDataset(
        X_test_raw, file_list=test_files, y=None,
        augment=False, max_frames=max_frames_train,
    )
    test_dataset.ch_means = norm_dataset.ch_means
    test_dataset.ch_stds = norm_dataset.ch_stds

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=12, persistent_workers=True,
        pin_memory=torch.cuda.is_available(),
    )

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

    test_file_names = [os.path.basename(f) for f in test_files]
    submit_df = pd.DataFrame({
        0: test_file_names,
        1: all_preds,
    })
    submit_df.to_csv(SUBMIT_FILE, index=False, header=False)
    print(f"提出ファイル作成完了: {SUBMIT_FILE} ({len(submit_df)} 件)")
    print(submit_df.head(10))

