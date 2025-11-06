# =============================================
# train_models.py - Fish Image Classification
# =============================================
import os, json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential, Model  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.applications import (  # type: ignore
    VGG16,
    ResNet50,
    MobileNetV2,
    InceptionV3,
    EfficientNetB0,
)

# ==============================
#  Paths & Params
# ==============================
base_path = r"C:\Users\abanu\Documents\Dataset\images.cv_jzk6llhf18tm3k0kyttxz\data"
train_dir, val_dir, test_dir = [
    os.path.join(base_path, p) for p in ["train", "val", "test"]
]
img_size, batch_size = (224, 224), 32
epochs_cnn, epochs_pre, fine_tune_epochs = 30, 30, 10
lr, os.makedirs("models", exist_ok=True)  # type: ignore

# ==============================
#  Data Generators
# ==============================
datagen_args = dict(
    rescale=1.0 / 255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)
train_gen = ImageDataGenerator(**datagen_args).flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical"
)
val_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical"
)
test_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
)

num_classes = len(train_gen.class_indices)
with open("models/class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f)
print(f"Number of fish classes: {num_classes}")


# ==============================
#  Utility Functions
# ==============================
def get_callbacks(path):
    return [
        ModelCheckpoint(path, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6),
    ]


def plot_history(histories, labels, title, filename):
    plt.figure(figsize=(12, 5))
    for h, lbl in zip(histories, labels):
        plt.plot(h.history["val_accuracy"], label=f"{lbl} Val Acc")
    plt.title(f"{title} Validation Accuracy")
    plt.legend()
    plt.savefig(filename)
    plt.close()


# ==============================
#  CNN Model
# ==============================
def build_cnn_model(input_shape=(224, 224, 3)):
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# ---- Train CNN ----
cnn = build_cnn_model()
history_cnn = cnn.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs_cnn,
    callbacks=get_callbacks("models/cnn_best_model.h5"),
)
test_loss, test_acc = cnn.evaluate(test_gen)
print(f"CNN Test Accuracy: {test_acc*100:.2f}%")

# ==============================
#  Pretrained Models
# ==============================
pretrained_models = {
    "vgg16": VGG16,
    "resnet50": ResNet50,
    "mobilenetv2": MobileNetV2,
    "inceptionv3": InceptionV3,
    "efficientnetb0": EfficientNetB0,
}


def build_transfer_model(model_class):
    base = model_class(
        weights="imagenet", include_top=False, input_shape=img_size + (3,)
    )
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    out = Dense(num_classes, activation="softmax")(x)
    model = Model(base.input, out)
    model.compile(
        optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model, base


best_model, best_acc, best_name = None, 0, ""

for name, model_class in pretrained_models.items():
    print(f"\n Training {name.upper()}...")
    model, base = build_transfer_model(model_class)
    ckpt = f"models/{name}_best_model.h5"
    base_hist = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_pre,
        callbacks=get_callbacks(ckpt),
    )

    # Fine-tuning
    print(f" Fine-tuning {name}...")
    base.trainable = True
    for layer in base.layers[:-50]:
        layer.trainable = False
    model.compile(
        optimizer=Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"]
    )
    fine_hist = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=fine_tune_epochs,
        callbacks=get_callbacks(ckpt),
    )

    # Evaluate & Track Best
    _, acc = model.evaluate(test_gen)
    print(f"{name} Accuracy: {acc*100:.2f}%")
    plot_history(
        [base_hist, fine_hist], ["Base", "Fine"], name, f"models/{name}_history.png"
    )
    if acc > best_acc:
        best_acc, best_name = acc, name
        model.save("models/best_fish_model.h5")

print(
    f"\n Best Model: {best_name} ({best_acc*100:.2f}%) saved as 'models/best_fish_model.h5'"
)
