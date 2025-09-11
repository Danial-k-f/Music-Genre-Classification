import numpy as np
import librosa
from src.data import extract_features, extract_spectrogram
from src.config import encoder
from src.data import prepare_spectrogram_dataset  # برای لود کردن lb

def predict_file_mfcc(model, file_path):
    """پیش‌بینی ژانر با مدل MFCC"""
    try:
        signal, sr = librosa.load(file_path, sr=22050, duration=30)
        features = extract_features(signal, sr)   # از data.py
        features = np.expand_dims(features, axis=0)  # (1, n_features)

        pred = model.predict(features)
        pred_class = np.argmax(pred, axis=1)[0]
        genre = encoder.inverse_transform([pred_class])[0]

        print(f"🎶 Predicted (MFCC Model) Genre: {genre}")
        return genre
    except Exception as e:
        print(f"❌ Prediction error (MFCC): {e}")
        return None


def predict_file_cnn(model, file_path):
    """پیش‌بینی ژانر با مدل CNN"""
    try:
        # اسپکتروگرام بساز
        spec = extract_spectrogram(file_path, max_pad_len=128)
        spec = np.expand_dims(spec, axis=-1)  # (H, W, 1)
        spec = np.expand_dims(spec, axis=0)   # (1, H, W, 1)

        # پیش‌بینی
        pred = model.predict(spec)
        pred_class = np.argmax(pred, axis=1)[0]

        # گرفتن لیبل‌ها از prepare_spectrogram_dataset
        (_, _, _, _), lb = prepare_spectrogram_dataset(test_size=0.2)
        genres = lb.classes_

        genre = genres[pred_class]
        print(f"🎶 Predicted (CNN Model) Genre: {genre}")
        return genre
    except Exception as e:
        print(f"❌ Prediction error (CNN): {e}")
        return None
