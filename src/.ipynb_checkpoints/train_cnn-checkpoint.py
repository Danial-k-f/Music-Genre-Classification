# src/train_cnn.py
import os
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from src.data import prepare_spectrogram_dataset

def build_vgg_cnn(input_shape, num_classes):
    model = models.Sequential()

    # Block 1 (فقط همینجا input_shape می‌گیره)
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.3))

    # Block 2
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.3))

    # Block 3
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.4))

    # Block 4
    model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.4))

    # Block 5
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.5))

    # Global pooling instead of Flatten
    model.add(layers.GlobalAveragePooling2D())

    # Dense layers
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_cnn_model():
    (X_train, X_val, y_train, y_val), lb = prepare_spectrogram_dataset(test_size=0.2)

    print(f"✅ Dataset loaded: {X_train.shape[0] + X_val.shape[0]} samples, {len(lb.classes_)} genres")
    print(f"🔍 y_train shape: {y_train.shape}, example: {y_train[:5]}")

    # چک کنیم که y one-hot هست یا نه
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        loss_fn = "categorical_crossentropy"
    else:
        loss_fn = "sparse_categorical_crossentropy"

    model = build_vgg_cnn(X_train.shape[1:], len(lb.classes_))
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint("models/cnn_model.keras", save_best_only=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks
    )

    os.makedirs("models", exist_ok=True)
    model.save("models/cnn_model_final.keras")
    print("✅ CNN Model saved at models/cnn_model_final.keras")

    return model, history

