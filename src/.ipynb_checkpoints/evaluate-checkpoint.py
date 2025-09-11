# src/evaluate.py
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from src.data import prepare_dataset, prepare_spectrogram_dataset

# 🔹 ارزیابی مدل MFCC
def evaluate_mfcc_model(model):
    X_train, X_test, y_train, y_test, lb = prepare_dataset()
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print("\n📊 Evaluation Report (MFCC Model):")
    print(classification_report(np.argmax(y_test, axis=1), y_pred_classes, target_names=lb.classes_))

# 🔹 ارزیابی مدل CNN
def evaluate_cnn_model(model):
    # دیتاست رو لود می‌کنیم (train/val split)
    (X_train, X_val, y_train, y_val), lb = prepare_spectrogram_dataset(test_size=0.2)

    print("📊 Evaluating CNN Model on validation set...")
    y_pred = model.predict(X_val)
    y_pred_classes = y_pred.argmax(axis=1)

    # اگر y_val به صورت One-Hot باشه → برگردون به لیبل عددی
    if y_val.ndim > 1 and y_val.shape[1] > 1:
        y_val_classes = np.argmax(y_val, axis=1)
    else:
        y_val_classes = y_val

    print(classification_report(y_val_classes, y_pred_classes, target_names=lb.classes_))