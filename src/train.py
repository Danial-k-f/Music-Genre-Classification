# src/train.py
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.data import prepare_dataset
from src.config import encoder

def train_model():
    (X_train, X_test, y_train, y_test), lb = prepare_dataset(test_size=0.2)

    model = Sequential([
        Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
        Dense(128, activation="relu"),
        Dense(len(lb.classes_), activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ModelCheckpoint("models/mfcc_model.keras", save_best_only=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=callbacks
    )

    os.makedirs("models", exist_ok=True)
    model.save("models/mfcc_model_final.keras")
    print("✅ MFCC Model saved at models/mfcc_model_final.keras")

    return model, history
