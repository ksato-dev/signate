"""
指定した .pth モデルでテストデータを推論し、submit.csv 形式の CSV を出力するスクリプト。

使い方:
  python predict_submit.py --pth best_model.pth [--output submit.csv]

train.py と同じデータ配置・config を前提とします。
4ステム（vocals / drums / bass / other）分離キャッシュと正規化パラメータ算出のため、学習用音声も参照します。
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import librosa
from natsort import natsorted
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import warnings
warnings.simplefilter('ignore')

import config

# config から取得
SEED = config.SEED
N_MELS = config.N_MELS
SR = config.SR
IMG_SIZE = config.IMG_SIZE
NUM_CLASSES = config.NUM_CLASSES
NUM_CHANNELS = config.NUM_CHANNELS
BATCH_SIZE = config.BATCH_SIZE
TEST_SIZE = config.TEST_SIZE
SEPARATION_CACHE_DIR = config.SEPARATION_CACHE_DIR
SEPARATION_MODEL = config.SEPARATION_MODEL
USE_AMP = getattr(config, 'USE_AMP', True)
MEL_CACHE_FILE = getattr(config, 'MEL_CACHE_FILE', 'data/precomputed_mel.npz')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

STEM_NAMES = ['vocals', 'drums', 'bass', 'other']


# ==============================================================================
# メルスペクトログラム
# ==============================================================================
def extract_melspectrogram(y, sr=SR, n_mels=None):
    n_fft = 2048
    hop_length = 512
    kwargs = {} if n_mels is None else {'n_mels': n_mels}
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, **kwargs)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def _load_3ch_from_cache(data):
    """train.py と同じ: キャッシュから (original, accompaniment, vocals) を返す。
    3ch モデル用に npz の original（元の音声）を使う必要がある。"""
    if 'drums' in data:
        vocals = data['vocals']
        drums = data['drums']
        bass = data['bass']
        other = data['other']
        accompaniment = drums + bass + other
        original = data['original'] if 'original' in data else (vocals + accompaniment)
        return original, accompaniment, vocals
    return data['original'], data['accompaniment'], data['vocals']


def load_and_pad_3ch(file_list, cache_dir=SEPARATION_CACHE_DIR, sr=SR, n_mels=N_MELS):
    """分離済みキャッシュから 3ch (original, accompaniment, vocals) メルスペクトログラムを読み込み、train.py と同一のデータで評価する。"""
    all_mels = []
    for f in tqdm(file_list, desc="  メルスペクトログラム抽出"):
        basename = os.path.splitext(os.path.basename(f))[0]
        cache_path = os.path.join(cache_dir, f"{basename}.npz")
        data = np.load(cache_path)
        original, accompaniment, vocals = _load_3ch_from_cache(data)
        data.close()
        mel_orig = extract_melspectrogram(original, sr=sr, n_mels=n_mels)
        mel_acc = extract_melspectrogram(accompaniment, sr=sr, n_mels=n_mels)
        mel_voc = extract_melspectrogram(vocals, sr=sr, n_mels=n_mels)
        min_frames = min(mel_orig.shape[1], mel_acc.shape[1], mel_voc.shape[1])
        mel_3ch = np.stack([
            mel_orig[:, :min_frames],
            mel_acc[:, :min_frames],
            mel_voc[:, :min_frames],
        ], axis=0)
        all_mels.append(mel_3ch)
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


def load_and_pad_4ch(file_list, cache_dir=SEPARATION_CACHE_DIR, sr=SR, n_mels=N_MELS):
    """分離済みキャッシュから4chメルスペクトログラムを読み込み、最大フレームでパディング"""
    all_mels = []
    for f in tqdm(file_list, desc="  メルスペクトログラム抽出"):
        basename = os.path.splitext(os.path.basename(f))[0]
        cache_path = os.path.join(cache_dir, f"{basename}.npz")
        data = np.load(cache_path)
        mel_stems = []
        for name in STEM_NAMES:
            mel = extract_melspectrogram(data[name], sr=sr, n_mels=n_mels)
            mel_stems.append(mel)
        data.close()
        min_frames = min(m.shape[1] for m in mel_stems)
        mel_stems = [m[:, :min_frames] for m in mel_stems]
        mel_4ch = np.stack(mel_stems, axis=0)
        all_mels.append(mel_4ch)
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


def _is_4stem_cache(cache_path):
    if not os.path.exists(cache_path):
        return False
    try:
        data = np.load(cache_path)
        result = 'drums' in data
        data.close()
        return result
    except Exception:
        return False


def load_mel_cache_if_valid(train_files, test_files, cache_path=None):
    """事前計算済みメルスペクトログラム（3ch）キャッシュがあればロードする。
    ファイルリストが一致する場合のみ有効。不一致なら None を返す。
    Returns:
        (X_all, X_test_raw, max_frames_train) or None if cache miss
    """
    cache_path = cache_path or MEL_CACHE_FILE
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


# ==============================================================================
# 音源分離（Demucs）4ステム
# ==============================================================================
def separate_audio_cached(file_path, demucs_model, cache_dir=SEPARATION_CACHE_DIR, sr=SR, device=DEVICE):
    from demucs.apply import apply_model
    basename = os.path.splitext(os.path.basename(file_path))[0]
    cache_path = os.path.join(cache_dir, f"{basename}.npz")
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        if 'drums' in data:
            stems = [data[n].copy() for n in STEM_NAMES]
            data.close()
            return tuple(stems)
        data.close()
        print(f"  旧キャッシュを検出: {basename} → 4stem で再分離します")
        os.remove(cache_path)
    os.makedirs(cache_dir, exist_ok=True)
    # 原音を目標サンプリングレートで読み込み
    original, _ = librosa.load(file_path, sr=sr, mono=True)
    demucs_sr = demucs_model.samplerate
    y_demucs, _ = librosa.load(file_path, sr=demucs_sr, mono=True)
    waveform = torch.FloatTensor(y_demucs).unsqueeze(0).repeat(2, 1)
    mix = waveform.unsqueeze(0)
    with torch.no_grad():
        sources = apply_model(demucs_model, mix, device=device)
    sources = sources.squeeze(0)
    source_names = demucs_model.sources
    stems_raw = {}
    for name in STEM_NAMES:
        idx = source_names.index(name)
        stems_raw[name] = sources[idx].mean(dim=0).cpu().numpy()
    stems = {}
    for name, wav in stems_raw.items():
        stems[name] = librosa.resample(wav, orig_sr=demucs_sr, target_sr=sr) if demucs_sr != sr else wav
    min_len = min(len(original), *(len(v) for v in stems.values()))
    original = original[:min_len]
    for name in stems:
        stems[name] = stems[name][:min_len]
    np.savez_compressed(cache_path, original=original, **stems)
    return tuple(stems[n] for n in STEM_NAMES)


def batch_separate(file_list, demucs_model, cache_dir=SEPARATION_CACHE_DIR, sr=SR, device=DEVICE):
    cached = sum(
        1 for f in file_list
        if _is_4stem_cache(os.path.join(cache_dir, f"{os.path.splitext(os.path.basename(f))[0]}.npz"))
    )
    if cached == len(file_list):
        print(f"  全{len(file_list)}ファイルがキャッシュ済み")
        return
    print(f"  キャッシュ済み: {cached}/{len(file_list)}, 残り{len(file_list) - cached}ファイルを分離中...")
    for f in tqdm(file_list, desc="  音源分離"):
        separate_audio_cached(f, demucs_model, cache_dir, sr, device)


# ==============================================================================
# Dataset（推論用・オーギュメントなし）
# ==============================================================================
class MelSpectrogramDatasetInference(Dataset):
    """推論用 Dataset。X のチャンネル数（3 or 4）に応じて正規化・リサイズ"""
    def __init__(self, X, ch_means, ch_stds, img_size=IMG_SIZE):
        self.X = X
        self.ch_means = np.asarray(ch_means)
        self.ch_stds = np.asarray(ch_stds)
        self.img_size = img_size
        self.n_ch = X.shape[1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        mel = self.X[idx].copy()
        for ch in range(self.n_ch):
            mel[ch] = (mel[ch] - self.ch_means[ch]) / (self.ch_stds[ch] + 1e-8)
        mel_tensor = torch.FloatTensor(mel)
        mel_tensor = torch.nn.functional.interpolate(
            mel_tensor.unsqueeze(0), size=(self.img_size, self.img_size),
            mode='bilinear', align_corners=False
        ).squeeze(0)
        return mel_tensor


# ==============================================================================
# モデル定義
# ==============================================================================
def build_model_4ch(num_classes=NUM_CLASSES, in_channels=NUM_CHANNELS):
    """現行 train.py と同じ: EfficientNet B7、4ch 入力、classifier 置換"""
    model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
    if in_channels != 3:
        old_conv = model.features[0][0]
        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3] = old_conv.weight.mean(dim=1)
            if old_conv.bias is not None:
                new_conv.bias = old_conv.bias
        model.features[0][0] = new_conv
    for param in model.features[:4].parameters():
        param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )
    return model


def build_model_legacy_3ch(num_classes=NUM_CLASSES):
    """旧形式: EfficientNet B7、3ch 入力、classifier 置換（拡張子なし）"""
    model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )
    return model


def _is_4ch_state_dict(state_dict):
    """state_dict が 4ch 入力（現行）かどうか"""
    if 'features.0.0.weight' not in state_dict:
        return False
    return state_dict['features.0.0.weight'].shape[1] == 4


def _is_legacy_3ch_state_dict(state_dict):
    """state_dict が 3ch レガシー（classifier のみ置換）かどうか"""
    keys = set(state_dict.keys())
    if 'classifier.1.weight' not in keys or 'classifier.4.weight' not in keys:
        return False
    if 'features.0.0.weight' in state_dict:
        return state_dict['features.0.0.weight'].shape[1] == 3
    return True


def build_model_for_checkpoint(state_dict, num_classes=NUM_CLASSES):
    """state_dict のキー・shape に応じてモデルを構築（4ch 現行 / 3ch レガシー）"""
    if _is_4ch_state_dict(state_dict):
        return build_model_4ch(num_classes=num_classes, in_channels=NUM_CHANNELS)
    if _is_legacy_3ch_state_dict(state_dict):
        return build_model_legacy_3ch(num_classes=num_classes)
    raise ValueError(
        "このチェックポイントは未対応です。4ch EfficientNet B7 または 3ch レガシーモデルのみ対応しています。"
    )


def convert_4ch_to_3ch(X_4ch):
    """4ch (vocals, drums, bass, other) を 3ch (original, accompaniment, vocals) に変換"""
    # original ≈ 全ステム和, accompaniment ≈ drums+bass+other, vocals
    ch0 = X_4ch[:, 0] + X_4ch[:, 1] + X_4ch[:, 2] + X_4ch[:, 3]
    ch1 = X_4ch[:, 1] + X_4ch[:, 2] + X_4ch[:, 3]
    ch2 = X_4ch[:, 0]
    return np.stack([ch0, ch1, ch2], axis=1)


# ==============================================================================
# メイン
# ==============================================================================
def _load_state_dict_from_pth(pth_path, device):
    """pth ファイルから state_dict を取り出す（checkpoint 形式と raw 形式の両方に対応）"""
    ckpt = torch.load(pth_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        return ckpt['model_state_dict'], True
    return ckpt, False


def main():
    parser = argparse.ArgumentParser(
        description='指定した pth で推論し submit.csv 形式の CSV を出力（複数指定でアンサンブル）')
    parser.add_argument('--pth', type=str, nargs='+', required=True,
                        help='モデル重みファイル (.pth)。複数指定で softmax 平均アンサンブル')
    parser.add_argument('--output', type=str, default='submit.csv', help='出力 CSV パス (default: submit.csv)')
    parser.add_argument('--no-separation', action='store_true',
                        help='音源分離をスキップ（既に data/separated にキャッシュがある場合）')
    args = parser.parse_args()

    for p in args.pth:
        if not os.path.exists(p):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {p}")

    n_models = len(args.pth)
    print(f"Device: {DEVICE}")
    print(f"モデル数: {n_models}")
    for i, p in enumerate(args.pth):
        print(f"  [{i+1}] {p}")
    print(f"出力: {args.output}")

    # データパス
    train_files = natsorted(glob.glob('data/train_sound_*/train_sound_*/train_*.au'))
    test_files = natsorted(glob.glob('data/test_sound_*/test_sound_*/test_*.au'))
    if not test_files:
        raise FileNotFoundError("テスト用音声が見つかりません (data/test_sound_*/test_sound_*/test_*.au)")
    print(f"学習ファイル数: {len(train_files)}, テストファイル数: {len(test_files)}")

    # 音源分離
    if not args.no_separation and train_files + test_files:
        print("\n=== 音源分離（Demucs） ===")
        from demucs.pretrained import get_model as get_demucs_model
        demucs_model = get_demucs_model(SEPARATION_MODEL)
        demucs_model.to(DEVICE)
        if train_files:
            print("学習データの音源分離:")
            batch_separate(train_files, demucs_model, SEPARATION_CACHE_DIR, SR, DEVICE)
        print("テストデータの音源分離:")
        batch_separate(test_files, demucs_model, SEPARATION_CACHE_DIR, SR, DEVICE)
        del demucs_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 最初のモデルでチャンネル種別を判定
    print("\n=== モデル種別の判定 ===")
    first_sd, _ = _load_state_dict_from_pth(args.pth[0], DEVICE)
    use_4ch = _is_4ch_state_dict(first_sd)
    print(f"  入力チャンネル: {'4ch (vocals/drums/bass/other)' if use_4ch else '3ch (legacy)'}")
    del first_sd

    # 学習データから max_frames と正規化パラメータを取得
    print("\n=== 正規化パラメータの算出（学習データ） ===")
    if not train_files:
        raise FileNotFoundError("正規化パラメータ算出のため学習用音声が必要です (data/train_sound_*/...)")

    mel_cache_path = MEL_CACHE_FILE
    cached_mel = None
    if not use_4ch:
        cached_mel = load_mel_cache_if_valid(train_files, test_files, mel_cache_path)
    if cached_mel is not None:
        X_all, X_test_raw, max_frames_train = cached_mel
        print(f"  事前計算メルキャッシュを使用: {mel_cache_path}")
        print(f"  X_all shape: {X_all.shape}, X_test shape: {X_test_raw.shape}, max_frames: {max_frames_train}")
    else:
        if use_4ch:
            X_all, max_frames_train = load_and_pad_4ch(train_files, SEPARATION_CACHE_DIR, SR, N_MELS)
        else:
            X_all, max_frames_train = load_and_pad_3ch(train_files, SEPARATION_CACHE_DIR, SR, N_MELS)

        print("\n=== テストデータ読み込み ===")
        if use_4ch:
            X_test_raw, max_frames_test = load_and_pad_4ch(test_files, SEPARATION_CACHE_DIR, SR, N_MELS)
        else:
            X_test_raw, max_frames_test = load_and_pad_3ch(test_files, SEPARATION_CACHE_DIR, SR, N_MELS)
        if max_frames_test > max_frames_train:
            X_test_raw = X_test_raw[:, :, :, :max_frames_train]
        elif max_frames_test < max_frames_train:
            pad_width = max_frames_train - max_frames_test
            X_test_raw = np.pad(
                X_test_raw,
                ((0, 0), (0, 0), (0, 0), (0, pad_width)),
                mode='constant',
                constant_values=X_test_raw.min()
            )
        print(f"X_test shape: {X_test_raw.shape}")

    n_ch = X_all.shape[1]
    ch_means = np.array([X_all[:, ch].mean() for ch in range(n_ch)])
    ch_stds = np.array([X_all[:, ch].std() for ch in range(n_ch)])
    print(f"max_frames_train: {max_frames_train}, チャンネル数: {n_ch}, ch_means.shape: {ch_means.shape}")

    test_dataset = MelSpectrogramDatasetInference(X_test_raw, ch_means, ch_stds, IMG_SIZE)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    amp_enabled = USE_AMP and torch.cuda.is_available()
    n_test = len(test_files)

    # 検証スコア（学習データを train/valid に分割し、valid で正解率を算出）
    print("\n=== 検証スコア ===")
    train_master = pd.read_csv('data/train_master.csv', index_col=0)
    y_all = train_master['label_id'].values
    indices = np.arange(len(X_all))
    _, valid_indices = train_test_split(
        indices, test_size=TEST_SIZE, random_state=SEED, stratify=y_all
    )
    X_valid = X_all[valid_indices]
    y_valid = y_all[valid_indices]
    valid_dataset = MelSpectrogramDatasetInference(X_valid, ch_means, ch_stds, IMG_SIZE)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    valid_ensemble_probs = np.zeros((len(y_valid), NUM_CLASSES), dtype=np.float64)
    ensemble_probs = np.zeros((n_test, NUM_CLASSES), dtype=np.float64)

    for mi, pth_path in enumerate(args.pth):
        print(f"\n=== モデル [{mi+1}/{n_models}] {pth_path} ===")
        state_dict, is_ckpt = _load_state_dict_from_pth(pth_path, DEVICE)
        model = build_model_for_checkpoint(state_dict, num_classes=NUM_CLASSES).to(DEVICE)
        model.load_state_dict(state_dict, strict=True)
        fmt = "checkpoint" if is_ckpt else "state_dict"
        print(f"  読み込み完了 ({fmt} 形式)")
        model.eval()

        # 検証
        batch_probs = []
        with torch.no_grad():
            for X_batch in valid_loader:
                X_batch = X_batch.to(DEVICE, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=amp_enabled):
                    outputs = model(X_batch)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                batch_probs.append(probs)
        model_valid_probs = np.concatenate(batch_probs, axis=0)
        valid_ensemble_probs += model_valid_probs
        model_valid_acc = accuracy_score(y_valid, model_valid_probs.argmax(axis=1))
        print(f"  単体 Valid Accuracy: {model_valid_acc:.4f}")

        # テスト推論
        batch_probs = []
        with torch.no_grad():
            for X_batch in tqdm(test_loader, desc="  推論"):
                X_batch = X_batch.to(DEVICE, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=amp_enabled):
                    outputs = model(X_batch)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                batch_probs.append(probs)
        ensemble_probs += np.concatenate(batch_probs, axis=0)

        del model, state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # アンサンブル結果
    valid_ensemble_probs /= n_models
    ensemble_probs /= n_models

    valid_preds = valid_ensemble_probs.argmax(axis=1)
    valid_acc = accuracy_score(y_valid, valid_preds)
    print(f"\nアンサンブル Valid Accuracy: {valid_acc:.4f} ({n_models} モデル, {len(valid_preds)} 件)")

    all_preds = ensemble_probs.argmax(axis=1)

    # CSV 出力
    test_file_names = [os.path.basename(f) for f in test_files]
    submit_df = pd.DataFrame({0: test_file_names, 1: all_preds})
    submit_df.to_csv(args.output, index=False, header=False)
    print(f"\n提出ファイル作成完了: {args.output} ({len(submit_df)} 件)")
    print(submit_df.head(10))


if __name__ == '__main__':
    main()
