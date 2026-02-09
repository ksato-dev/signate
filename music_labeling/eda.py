# 使用ライブラリ
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import librosa
import librosa.display
import IPython.display as ipd

import glob
from natsort import natsorted

from sklearn.decomposition import PCA
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split

import warnings
warnings.simplefilter('ignore')

import pandas as pd
import librosa
import librosa.display

# メタデータの読み込み
train_master = pd.read_csv('data/train_master.csv', index_col=0)
label_master = pd.read_csv('data/label_master.csv')
sample_submit = pd.read_csv('data/sample_submit.csv', header=None)

# 最初の音声データ(train_0.au)の見てみる
y, sr = librosa.load('data/train_sound_1/train_sound_1/train_0.au')
print('file size : ', y.shape)
print('sampling rate : ', sr)
print('len of audio : ', y.shape[0]/sr)

ipd.Audio(y, rate=sr)

# ラベルリストの作成
label_list = label_master.to_dict()['label_name']
print(label_list)

# train にアクセス
label_id = train_master['label_id'].iloc[0]
print(label_id)
print(label_list[label_id])
print(train_master['label_id'].value_counts().sort_index())

# # ラベルの分布を可視化
# plt.figure(figsize=(10, 6))
# train_master['label_id'].value_counts().sort_index().plot(kind='bar')
# plt.title('Label Distribution')
# plt.xlabel('Label ID')
# plt.ylabel('Count')
# plt.show()

# 音声波形の表示
# plt.figure(figsize=(15, 5))
# librosa.display.waveshow(y=y, sr=sr)
# plt.title(f'label name : {label_list[label_id]}')
# plt.show()

# フーリエ変換した波形の表示
fft_data = np.abs(librosa.stft(y))
# plt.figure(figsize=(15, 5))
# librosa.display.waveshow(y=fft_data, sr=sr)
# plt.title(f'label name : {label_list[label_id]}')
# plt.show()

# 人間の聴覚に近い変換をするメルスペクトログラムの表示
mel = librosa.feature.melspectrogram(y=y, sr=sr)
mel_db = librosa.amplitude_to_db(mel, ref=np.max)
# plt.figure(figsize=(15, 5))
# librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='log', cmap='cool')
# plt.colorbar()
# plt.title(f'MelSpectrogram : {label_list[label_id]}')
# plt.show()

# メルスペクトログラムで特徴量を抽出
import glob
from natsort import natsorted

train_files = natsorted(glob.glob('data/train_sound_*/train_sound_*/train_*.au'))
train = []
for file in train_files:
    y, sr = librosa.load(file)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.amplitude_to_db(mel, ref=np.max)  # shape: (n_mels=128, time_frames)
    train.append(mel_db)

# 各メルスペクトログラムの形状を確認
time_frames = [t.shape[1] for t in train]
print(f"メルスペクトログラムの形状:")
print(f"  n_mels: {train[0].shape[0]}")
print(f"  time_frames 最小値: {min(time_frames)}")
print(f"  time_frames 最大値: {max(time_frames)}")

# 最大フレーム数に合わせてゼロパディング（2D形状を維持）
max_frames = max(time_frames)
train_padded = []
for mel_db in train:
    pad_width = max_frames - mel_db.shape[1]
    # 右側をゼロ（無音相当の最小dB値）でパディング
    padded = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=mel_db.min())
    train_padded.append(padded)

# numpy配列に変換: (n_samples, n_mels, max_frames)
X = np.array(train_padded)
y_labels = train_master['label_id'].values
print(f"\n特徴量の形状: {X.shape}")   # (500, 128, max_frames)
print(f"ラベルの形状: {y_labels.shape}")  # (500,)
print(f"クラス数: {len(np.unique(y_labels))}")

# CNN用に次元追加: (n_samples, 1, n_mels, max_frames) = (batch, channel, height, width)
X = X[:, np.newaxis, :, :]
print(f"CNN入力形状: {X.shape}")

# train/valid 分割
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y_labels, test_size=0.1, random_state=0, stratify=y_labels
)
print(f"\nX_train: {X_train.shape}, X_valid: {X_valid.shape}")
print(f"y_train: {y_train.shape}, y_valid: {y_valid.shape}")

# 分類器の実装