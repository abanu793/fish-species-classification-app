# ==============================
# evaluate_models.py - Fish Classification Evaluation
# ==============================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore

# -----------------------------
#  Setup
# -----------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
test_dir = r"C:\Users\abanu\Documents\Dataset\images.cv_jzk6llhf18tm3k0kyttxz\data\test"
models_dir = "models"
img_size, batch_size = (224, 224), 32

# -----------------------------
#  Data Generator
# -----------------------------
test_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
)
class_labels = list(test_gen.class_indices.keys())


# -----------------------------
#  Helper Functions
# -----------------------------
def evaluate_model(model_path):
    """Load, evaluate, and plot confusion matrix for a model."""
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

    preds = np.argmax(model.predict(test_gen, verbose=1), axis=1)
    y_true = test_gen.classes
    acc = np.mean(preds == y_true)

    report = classification_report(y_true, preds, target_names=class_labels)
    cm = confusion_matrix(y_true, preds)
    plot_confusion_matrix(cm, os.path.basename(model_path))
    return acc, report


def plot_confusion_matrix(cm, model_name):
    """Save confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()


# -----------------------------
#  Evaluate All Models
# -----------------------------
model_files = [f for f in os.listdir(models_dir) if f.endswith(".h5")]
if not model_files:
    raise SystemExit(" No model files found in 'models/'. Please train models first.")

results = {}
for model_file in model_files:
    print(f"\n Evaluating: {model_file}")
    acc, report = evaluate_model(os.path.join(models_dir, model_file))
    print(f" Accuracy: {acc*100:.2f}%\n{report}")
    results[model_file] = {"accuracy": acc, "classification_report": report}

# -----------------------------
#  Save Summaries
# -----------------------------
summary_txt = os.path.join(models_dir, "evaluation_summary.txt")
with open(summary_txt, "w") as f:
    for name, m in results.items():
        f.write(f"Model: {name}\nAccuracy: {m['accuracy']*100:.2f}%\n")
        f.write(m["classification_report"] + "\n" + "=" * 60 + "\n")

summary_csv = os.path.join(models_dir, "evaluation_summary.csv")
pd.DataFrame(
    [{"Model": n, "Accuracy": m["accuracy"]} for n, m in results.items()]
).sort_values("Accuracy", ascending=False).to_csv(summary_csv, index=False)

# -----------------------------
#  Best Model
# -----------------------------
best_name, best_metrics = max(results.items(), key=lambda x: x[1]["accuracy"])
print(f"\n Best model: {best_name} ({best_metrics['accuracy']*100:.2f}%)")
print(f" Text summary: {summary_txt}")
print(f" CSV summary:  {summary_csv}")
print(f" Confusion matrices in: {models_dir}/")
