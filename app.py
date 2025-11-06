# ==============================
# app.py - Fish Species Classification Streamlit App
# ==============================
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os, json, pandas as pd

# ------------------------------
#  App Setup
# ------------------------------
st.set_page_config(page_title=" Fish Species Classification", layout="centered")
st.title(" Fish Species Classification App")

MODELS_DIR = "models"
IMG_SIZE = (224, 224)

# ------------------------------
#  Load Model Files + Accuracy
# ------------------------------
model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".h5")]
if not model_files:
    st.error("No models found in the 'models/' folder. Please train models first.")
    st.stop()

# Optional: Load evaluation summary (if exists)
acc_map = {}
summary_csv = os.path.join(MODELS_DIR, "evaluation_summary.csv")
if os.path.exists(summary_csv):
    df = pd.read_csv(summary_csv)
    acc_map = dict(zip(df["Model"], df["Accuracy"] * 100))


def label_with_acc(name):
    return f"{name} ({acc_map[name]:.2f}%)" if name in acc_map else name


display_names = [label_with_acc(f) for f in model_files]

default_model = (
    "best_fish_model.h5" if "best_fish_model.h5" in model_files else model_files[0]
)
model_choice = st.sidebar.selectbox(
    "Select Model", display_names, index=model_files.index(default_model)
)
model_file = model_files[display_names.index(model_choice)]


# ------------------------------
#  Load Model (cached)
# ------------------------------
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)


model = load_model(os.path.join(MODELS_DIR, model_file))
st.success(f" Loaded model: {model_file}")

# ------------------------------
#  Load Class Indices
# ------------------------------
idx_to_class = {}
path = os.path.join(MODELS_DIR, "class_indices.json")
if os.path.exists(path):
    with open(path) as f:
        data = json.load(f)
        idx_to_class = {v: k for k, v in data.items()}

# ------------------------------
#  Image Upload & Prediction
# ------------------------------
file = st.file_uploader(" Upload a fish image", type=["jpg", "jpeg", "png"])
if not file:
    st.stop()

image = Image.open(file).convert("RGB")
st.image(image, caption="Uploaded Image", use_container_width=True)

# Preprocess
img = np.expand_dims(np.array(image.resize(IMG_SIZE)) / 255.0, axis=0)

# Predict
preds = model.predict(img)
idx, conf = np.argmax(preds), np.max(preds)
cls = idx_to_class.get(idx, str(idx))

# Display
st.subheader(f" Predicted: {cls}")
st.write(f"Confidence: {conf*100:.2f}%")

# All class probabilities
st.subheader(" Class Probabilities")
st.bar_chart({idx_to_class.get(i, str(i)): float(p) for i, p in enumerate(preds[0])})
