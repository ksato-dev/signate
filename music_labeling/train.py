"""
音楽ジャンル分類 - EfficientNet B7 (PyTorch)
メルスペクトログラムを画像として扱い、事前学習済みEfficientNet B7でFine-tuning
"""

import os
import glob
import signal
import sys
import numpy as np
import pandas as pd
import librosa
from natsort import natsorted

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import warnings
warnings.simplefilter('ignore')

# ==============================================================================
# 設定
# ==============================================================================
SEED = 42
N_MELS = 128
SR = 22050
IMG_SIZE = 600           # EfficientNet B7 の標準入力サイズ
BATCH_SIZE = 24          # RTX3090 (24GB) なら16-24が可能。メモリ不足時は下げる
EPOCHS = 50
LR = 2e-4                # バッチサイズ増加に伴い学習率も調整（Linear Scaling: 4→16で4倍）
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 10
TEST_SIZE = 0.1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Device: {DEVICE}")

# ==============================================================================
# データ読み込み・メルスペクトログラム抽出
# ==============================================================================
def extract_melspectrogram(file_path, sr=SR, n_mels=None):
    """音声ファイルからメルスペクトログラム(dB)を抽出
    
    Args:
        n_mels: メルバンド数。Noneの場合はlibrosaのデフォルト値(128)を使用
    """
    y, sr = librosa.load(file_path, sr=sr)
    # n_melsがNoneの場合はデフォルト値を使用（128）
    kwargs = {} if n_mels is None else {'n_mels': n_mels}
    mel = librosa.feature.melspectrogram(y=y, sr=sr, **kwargs)
    mel_db = librosa.amplitude_to_db(mel, ref=np.max)
    return mel_db


def load_and_pad(file_list, sr=SR, n_mels=N_MELS):
    """音声ファイル群からメルスペクトログラムを抽出し、パディングして統一サイズにする"""
    spectrograms = []
    for f in file_list:
        mel_db = extract_melspectrogram(f, sr=sr, n_mels=n_mels)
        spectrograms.append(mel_db)

    # 最大フレーム数に合わせてパディング
    max_frames = max(s.shape[1] for s in spectrograms)
    padded = []
    for s in spectrograms:
        pad_width = max_frames - s.shape[1]
        p = np.pad(s, ((0, 0), (0, pad_width)), mode='constant', constant_values=s.min())
        padded.append(p)

    return np.array(padded), max_frames


print("=== 学習データの読み込み ===")
train_master = pd.read_csv('data/train_master.csv', index_col=0)
label_master = pd.read_csv('data/label_master.csv')

# メル周波数バンドの実際の周波数分布を確認
mel_freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=0.0, fmax=SR/2.0)
print(f"\nメル周波数バンドの周波数分布:")
print(f"  総バンド数: {N_MELS}")
print(f"  周波数範囲: {mel_freqs[0]:.1f} Hz ～ {mel_freqs[-1]:.1f} Hz")
print(f"  最初の10バンド: {mel_freqs[:10]}")
print(f"  最後の10バンド: {mel_freqs[-10:]}")
print(f"  3000Hz付近のバンド: {np.where((mel_freqs >= 2800) & (mel_freqs <= 3200))[0]}")
print(f"  → 3000Hzは約 {np.argmin(np.abs(mel_freqs - 3000))} 番目のバンド")

train_files = natsorted(glob.glob('data/train_sound_*/train_sound_*/train_*.au'))
print(f"\n学習ファイル数: {len(train_files)}")

X_all, max_frames_train = load_and_pad(train_files)
y_all = train_master['label_id'].values
print(f"X_all shape: {X_all.shape}")  # (500, 128, max_frames)
print(f"y_all shape: {y_all.shape}")

print("\n=== テストデータの読み込み ===")
test_files = natsorted(glob.glob('data/test_sound_*/test_sound_*/test_*.au'))
print(f"テストファイル数: {len(test_files)}")

X_test_raw, max_frames_test = load_and_pad(test_files)
# テストデータのフレーム数を学習データに合わせる
if max_frames_test > max_frames_train:
    X_test_raw = X_test_raw[:, :, :max_frames_train]
elif max_frames_test < max_frames_train:
    pad_width = max_frames_train - max_frames_test
    X_test_raw = np.pad(X_test_raw, ((0, 0), (0, 0), (0, pad_width)),
                        mode='constant', constant_values=X_test_raw.min())
print(f"X_test shape: {X_test_raw.shape}")

# ==============================================================================
# Dataset
# ==============================================================================
class MelSpectrogramDataset(Dataset):
    """メルスペクトログラムを3チャネル画像に変換して返すDataset"""

    def __init__(self, X, y=None, img_size=IMG_SIZE, augment=False):
        self.X = X  # (N, n_mels, time_frames)
        self.y = y
        self.img_size = img_size
        self.augment = augment

        # 正規化パラメータ（全データから計算）
        self.mean = X.mean()
        self.std = X.std()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        mel = self.X[idx].copy()  # (n_mels, time_frames)

        # 標準化
        mel = (mel - self.mean) / (self.std + 1e-8)

        # Data Augmentation（学習時のみ）
        if self.augment:
            # Time masking: ランダムに時間方向の一部をマスク
            if np.random.random() < 0.5:
                t = mel.shape[1]
                t_mask = np.random.randint(0, max(1, t // 10))
                t_start = np.random.randint(0, max(1, t - t_mask))
                mel[:, t_start:t_start + t_mask] = 0

            # Frequency masking: ランダムに周波数方向の一部をマスク
            if np.random.random() < 0.5:
                f = mel.shape[0]
                f_mask = np.random.randint(0, max(1, f // 10))
                f_start = np.random.randint(0, max(1, f - f_mask))
                mel[f_start:f_start + f_mask, :] = 0

        # Tensor に変換 (1, H, W)
        mel_tensor = torch.FloatTensor(mel).unsqueeze(0)

        # リサイズ (1, H, W) -> (1, img_size, img_size)
        mel_tensor = torch.nn.functional.interpolate(
            mel_tensor.unsqueeze(0), size=(self.img_size, self.img_size),
            mode='bilinear', align_corners=False
        ).squeeze(0)

        # 1ch -> 3ch（EfficientNet は3チャネル入力）
        mel_tensor = mel_tensor.repeat(3, 1, 1)  # (3, img_size, img_size)

        if self.y is not None:
            label = torch.LongTensor([self.y[idx]]).squeeze()
            return mel_tensor, label
        else:
            return mel_tensor


# ==============================================================================
# Train / Valid 分割
# ==============================================================================
X_train, X_valid, y_train, y_valid = train_test_split(
    X_all, y_all, test_size=TEST_SIZE, random_state=SEED, stratify=y_all
)
print(f"\nX_train: {X_train.shape}, X_valid: {X_valid.shape}")

# 正規化パラメータは学習データから計算して共有
train_dataset = MelSpectrogramDataset(X_train, y_train, augment=True)
valid_dataset = MelSpectrogramDataset(X_valid, y_valid, augment=False)
valid_dataset.mean = train_dataset.mean  # 学習データの統計量を使用
valid_dataset.std = train_dataset.std

test_dataset = MelSpectrogramDataset(X_test_raw, y=None, augment=False)
test_dataset.mean = train_dataset.mean
test_dataset.std = train_dataset.std

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
# モデル定義: EfficientNet B7
# ==============================================================================
def build_model(num_classes=NUM_CLASSES):
    """事前学習済み EfficientNet B7 を Fine-tuning 用に構築"""
    model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)

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

        optimizer.zero_grad(set_to_none=True)  # メモリ効率向上
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        # メモリリーク対策: 計算結果をCPUに移してから削除
        total_loss += loss.detach().item() * len(y_batch)
        preds = outputs.detach().argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.cpu().numpy())

        # 明示的にメモリ解放
        del outputs, loss, X_batch, y_batch

        # 定期的にキャッシュクリア（100バッチごと）
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

        # 明示的にメモリ解放
        del outputs, loss, X_batch, y_batch

    # 評価後はキャッシュクリア
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


# ==============================================================================
# チェックポイント管理
# ==============================================================================
CHECKPOINT_FILE = 'checkpoint.pth'
BEST_MODEL_FILE = 'best_model.pth'

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


def load_checkpoint(model, optimizer, scheduler):
    """チェックポイントから学習を再開"""
    if os.path.exists(CHECKPOINT_FILE):
        print(f"チェックポイントを読み込み: {CHECKPOINT_FILE}")
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_valid_acc = checkpoint['best_valid_acc']
        best_epoch = checkpoint['best_epoch']
        patience_counter = checkpoint['patience_counter']
        print(f"  再開: epoch {start_epoch} から")
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

# チェックポイントから再開（あれば）
start_epoch, best_valid_acc, best_epoch, patience_counter = load_checkpoint(model, optimizer, scheduler)
patience = 10

current_epoch = start_epoch
try:
    for epoch in range(start_epoch, EPOCHS + 1):
        current_epoch = epoch
        
        if interrupted:
            print("\n学習を中断します...")
            save_checkpoint(epoch - 1, model, optimizer, scheduler, best_valid_acc, best_epoch, patience_counter)
            break

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Valid Loss: {valid_loss:.4f} Acc: {valid_acc:.4f} | "
              f"LR: {lr:.6f}")

        # ベストモデル保存
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_FILE)
            # ベストモデル更新時のみチェックポイントも保存
            save_checkpoint(epoch, model, optimizer, scheduler, best_valid_acc, best_epoch, patience_counter)
            print(f"  → ベストモデル更新 (Valid Acc: {best_valid_acc:.4f})")
        else:
            patience_counter += 1

        # エポック終了後にメモリクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Early Stopping
        if patience_counter >= patience:
            print(f"\nEarly Stopping at epoch {epoch} (ベスト: epoch {best_epoch})")
            break

    print(f"\n=== 学習完了 ===")
    print(f"ベストモデル: epoch {best_epoch}, Valid Acc: {best_valid_acc:.4f}")

except KeyboardInterrupt:
    print("\n\nCtrl+Cで中断されました。チェックポイントを保存します...")
    save_checkpoint(current_epoch - 1, model, optimizer, scheduler, best_valid_acc, best_epoch, patience_counter)
    print("チェックポイント保存完了。次回は同じスクリプトで再開できます。")

finally:
    # GPUメモリをクリア
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
        
        # メモリ解放
        del outputs, X_batch

# 推論後もメモリクリア
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 提出CSV作成
test_file_names = [os.path.basename(f) for f in test_files]
submit_df = pd.DataFrame({
    0: test_file_names,
    1: all_preds
})
submit_df.to_csv('submit.csv', index=False, header=False)
print(f"提出ファイル作成完了: submit.csv ({len(submit_df)} 件)")
print(submit_df.head(10))
