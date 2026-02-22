"""
CNN Training Script — MobileNetV2 Transfer Learning
-----------------------------------------------------
Place your dataset in:
    dataset/
        fresh/      ← images of fresh food
        expired/    ← images of expired food

Run:
    python model/train.py

The trained model is saved as  model/food_freshness_model.h5
"""

import os
import sys

# Add project root to path so this can be run from anywhere
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ── Config ──────────────────────────────────────
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "food_freshness_model.h5")


def build_model() -> Model:
    """Build MobileNetV2-based binary classifier."""
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))

    # Freeze base layers
    for layer in base.layers:
        layer.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)  # 0 = expired, 1 = fresh

    model = Model(inputs=base.input, outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def get_data_generators():
    """Create train/validation generators with augmentation."""
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        validation_split=0.2,
    )

    train_data = train_gen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        classes=["expired", "fresh"],   # 0 = expired, 1 = fresh
    )

    val_data = train_gen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        classes=["expired", "fresh"],
    )

    return train_data, val_data


def train():
    """Train the model and save weights."""
    if not os.path.isdir(DATASET_DIR):
        print(f"❌  Dataset directory not found: {DATASET_DIR}")
        print("   Create  dataset/fresh/  and  dataset/expired/  with your images.")
        sys.exit(1)

    fresh_dir = os.path.join(DATASET_DIR, "fresh")
    expired_dir = os.path.join(DATASET_DIR, "expired")
    if not os.path.isdir(fresh_dir) or not os.path.isdir(expired_dir):
        print("❌  Expected sub-folders:  dataset/fresh/  and  dataset/expired/")
        sys.exit(1)

    print("🔧  Building model …")
    model = build_model()
    model.summary()

    print("📂  Loading dataset …")
    train_data, val_data = get_data_generators()
    print(f"   Training samples : {train_data.samples}")
    print(f"   Validation samples: {val_data.samples}")

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor="val_accuracy"),
    ]

    print("🚀  Training …")
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    model.save(MODEL_SAVE_PATH)
    print(f"✅  Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
