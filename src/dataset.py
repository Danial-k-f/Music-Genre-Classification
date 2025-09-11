import os
import numpy as np
import librosa
import random
from src.config import DATA_PATH, encoder

def extract_features(signal, sr):
    """استخراج ویژگی‌های ترکیبی از هر فایل (MFCC + Chroma + Spectral Contrast)"""
    # MFCCs
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)

    # Chromagram
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
    chroma_scaled = np.mean(chroma.T, axis=0)

    # Spectral Contrast
    spec_contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)
    spec_contrast_scaled = np.mean(spec_contrast.T, axis=0)

    # 🔹 ترکیب همه ویژگی‌ها در یک وکتور
    return np.hstack([mfcc_scaled, chroma_scaled, spec_contrast_scaled])


def augment_signal(signal, sr):
    """اعمال Augmentation ساده روی سیگنال"""
    choice = random.choice(["pitch", "stretch", "noise", "none"])
    if choice == "pitch":
        return librosa.effects.pitch_shift(
            signal, sr=sr, n_steps=random.choice([-2, -1, 1, 2])
        )
    elif choice == "stretch":
        rate = random.uniform(0.8, 1.2)
        try:
            return librosa.effects.time_stretch(signal, rate=rate)
        except TypeError:
            return librosa.effects.time_stretch(signal, rate)
    elif choice == "noise":
        noise = np.random.normal(0, 0.005, signal.shape)
        return signal + noise
    else:
        return signal


def load_audio_files(max_files_per_genre=None, augment=False):
    """بارگذاری فایل‌های صوتی و استخراج ویژگی‌ها"""
    X, y = [], []
    genres = encoder.classes_  # 🔹 از encoder مرکزی استفاده می‌کنیم

    for genre in genres:
        genre_folder = os.path.join(DATA_PATH, genre)
        count = 0

        for file in os.listdir(genre_folder):
            if file.endswith(".wav") or file.endswith(".au"):
                file_path = os.path.join(genre_folder, file)
                signal, sr = librosa.load(file_path, sr=22050, duration=30)

                # Augmentation اختیاری
                if augment:
                    signal = augment_signal(signal, sr)

                features = extract_features(signal, sr)
                X.append(features)
                y.append(genre)

                count += 1
                if max_files_per_genre and count >= max_files_per_genre:
                    break

    return np.array(X), np.array(y)
