"""Train a CNN symbol classifier for the Handwritten Equation Solver project."""
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA_DIR = Path("dataset")
MODEL_DIR = Path("models")
IMG_SIZE = 45
SEED = 42

# Dataset folders must use these exact names (one folder per class).
LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/", "(", ")"]


def load_dataset():
    images, labels = [], []
    for label_index, label in enumerate(LABELS):
        folder = DATA_DIR / label
        if not folder.exists():
            print(f"Skipping missing class folder: {folder}")
            continue
        for file in folder.rglob("*"):
            if file.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                continue
            image = cv2_read_grayscale(file)
            if image is None:
                continue
            image = normalize_symbol_image(image)
            images.append(image)
            labels.append(label_index)
    if not images:
        raise RuntimeError("No images found. Add symbol images to dataset/<class-name>/ folders.")
    return np.asarray(images, dtype=np.float32)[..., None], np.asarray(labels, dtype=np.int64)


def cv2_read_grayscale(path):
    import cv2
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def normalize_symbol_image(image):
    """Resize while preserving aspect ratio, center on a 45x45 canvas, and normalize."""
    import cv2
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    # Most datasets have dark ink on a light background; invert to white ink on black.
    if image.mean() > 127:
        image = 255 - image
    return image.astype(np.float32) / 255.0


def build_model(num_classes):
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    import cv2  # used by preprocessing
    MODEL_DIR.mkdir(exist_ok=True)
    X, y = load_dataset()
    # Stratified split, keeping a test set aside for final evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y
    )
    model = build_model(len(LABELS))
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(MODEL_DIR / "best_model.keras", monitor="val_loss", save_best_only=True),
    ]
    history = model.fit(
        X_train, y_train, validation_split=0.1, epochs=30, batch_size=64,
        callbacks=callbacks, verbose=1
    )
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Held-out test accuracy: {test_acc:.4f}")
    predictions = model.predict(X_test, verbose=0).argmax(axis=1)
    print(classification_report(y_test, predictions, labels=list(range(len(LABELS))),
                             target_names=LABELS, zero_division=0))
    model.save(MODEL_DIR / "equation_symbol_model.keras")
    np.savez(MODEL_DIR / "training_history.npz",
             accuracy=history.history.get("accuracy", []),
             val_accuracy=history.history.get("val_accuracy", []),
             loss=history.history.get("loss", []),
             val_loss=history.history.get("val_loss", []))
    print("Saved model to models/equation_symbol_model.keras")
