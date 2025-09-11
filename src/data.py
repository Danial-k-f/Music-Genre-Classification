# src/data.py
import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

DATA_PATH = "data/gtzan_dataset/genres"
MAX_LEN = 128  # طول استاندارد اسپکتروگرام (سطر زمان)

# ------------------------------
# آماده‌سازی دیتاست با MFCC
# ------------------------------
def extract_features(signal, sr):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

def prepare_dataset(test_size=0.2):
    X, y = [], []
    genres = os.listdir(DATA_PATH)
    for genre in genres:
        genre_folder = os.path.join(DATA_PATH, genre)
        if not os.path.isdir(genre_folder):
            continue
        for file in os.listdir(genre_folder):
            if file.endswith(".wav") or file.endswith(".au"):
                file_path = os.path.join(genre_folder, file)
                signal, sr = librosa.load(file_path, sr=22050, duration=30)
                features = extract_features(signal, sr)
                X.append(features)
                y.append(genre)

    X = np.array(X)
    lb = LabelBinarizer()
    y = lb.fit_transform(y)

    return train_test_split(X, y, test_size=test_size, random_state=42), lb

# ------------------------------
# آماده‌سازی دیتاست با Spectrogram
# ------------------------------
def extract_spectrogram(file_path, max_pad_len=MAX_LEN):
    """تولید و پد/برش اسپکتروگرام از یک فایل"""
    signal, sr = librosa.load(file_path, sr=22050, duration=30)
    spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
    spec_db = librosa.power_to_db(spec, ref=np.max)

    if spec_db.shape[1] < max_pad_len:
        pad_width = max_pad_len - spec_db.shape[1]
        spec_db = np.pad(spec_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        spec_db = spec_db[:, :max_pad_len]

    return spec_db

def prepare_spectrogram_dataset(test_size=0.2, max_len=MAX_LEN):
    X, y = [], []
    genres = os.listdir(DATA_PATH)
    for genre in genres:
        genre_folder = os.path.join(DATA_PATH, genre)
        if not os.path.isdir(genre_folder):
            continue
        for file in os.listdir(genre_folder):
            if file.endswith(".wav") or file.endswith(".au"):
                file_path = os.path.join(genre_folder, file)
                spec = extract_spectrogram(file_path, max_pad_len=max_len)
                X.append(spec)
                y.append(genre)

    X = np.array(X)[..., np.newaxis]  # اضافه کردن بعد کانال
    lb = LabelBinarizer()
    y = lb.fit_transform(y)

    return train_test_split(X, y, test_size=test_size, random_state=42), lb
