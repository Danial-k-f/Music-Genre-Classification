import librosa
import numpy as np

def extract_features(signal, sr=22050, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    mfccs = mfccs.mean(axis=1)
    return mfccs
