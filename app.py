# app.py
import streamlit as st
from PIL import Image
import numpy as np
import io
import time
import os

# --- ML libs ---
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# PyTorch (DINOv2 embedding)
import torch
import torchvision.transforms as T

# Utilities
import pandas as pd

st.set_page_config(layout="wide", page_title="POC: MobileNetV2 vs DINOv2")

# ---------------------------
# Helpers: load models
# ---------------------------

@st.cache_resource(show_spinner=False)
def load_mobilenet_model(path="best_mobilenetv2_finetuned.keras"):
    if not os.path.exists(path):
        st.error(f"MobileNet model not found at {path}")
        return None
    model = load_model(path)
    return model

@st.cache_resource(show_spinner=False)
def load_dinov2_model(dino_classifier_path="best_dinov2_classifier.keras"):
    # load classifier (Keras) trained on embeddings
    if not os.path.exists(dino_classifier_path):
        st.error(f"DINOv2 classifier not found at {dino_classifier_path}")
        return None, None
    clf = load_model(dino_classifier_path)

    # load DINOv2 embedder (PyTorch) via hub
    # we use try/except because network/availability may vary
    try:
        # choose s14 variant for moderate size; change if you used another
        dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14') 
        dinov2.eval()
    except Exception as e:
        st.warning(f"Impossible de charger DINOv2 via torch.hub: {e}")
        dinov2 = None
    return dinov2, clf

# ---------------------------
# Embedding extraction
# ---------------------------

# Use transforms matching training
dinov2_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

def image_to_mobilenet_input(img: Image.Image, target_size=(224,224)):
    img = img.convert("RGB").resize(target_size)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, 0).astype(np.float32)
    return arr

def image_to_dinov2_embedding(img: Image.Image, dinov2_model, device='cpu'):
    """
    Returns a numpy vector embedding extracted by DINOv2 (flattened)
    """
    if dinov2_model is None:
        raise RuntimeError("DINOv2 model not loaded")
    img_t = dinov2_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = dinov2_model(img_t).cpu().numpy()
    return emb.reshape(emb.shape[0], -1)  # shape (1, D)

# ---------------------------
# UI layout
# ---------------------------
st.title("Proof of Concept — MobileNetV2 vs DINOv2")
st.markdown("Upload an image to compare predictions from MobileNetV2 (Keras) and DINOv2 (embedder + classifier).")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Chargement des modèles")
    mobilenet_path = st.text_input("Path MobileNetV2 Keras model", "best_mobilenetv2_finetuned.keras")
    dino_classifier_path = st.text_input("Path DINOv2 classifier (Keras)", "best_dinov2_classifier.keras")
    load_button = st.button("Charger les modèles")
    if load_button:
        with st.spinner("Loading models..."):
            mobilenet = load_mobilenet_model(mobilenet_path)
            dinov2_embedder, dino_clf = load_dinov2_model(dino_classifier_path)
            st.success("Modèles chargés (si pas d'erreur ci-dessus).")
    else:
        # Provide models preloaded if cached
        mobilenet = load_mobilenet_model(mobilenet_path)
        dinov2_embedder, dino_clf = load_dinov2_model(dino_classifier_path)

with col2:
    st.header("Upload / Exemple")
    uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    st.write("Ou tester un exemple du dataset :")
    # If you packaged sample images in a folder 'samples/', list them
    sample_folder = "samples"
    sample_files = []
    if os.path.exists(sample_folder):
        sample_files = [os.path.join(sample_folder, f) for f in os.listdir(sample_folder) if f.lower().endswith(('.jpg','.png','.jpeg'))]
    if sample_files:
        sample_choice = st.selectbox("Samples", ["--"] + sample_files)
        if sample_choice and sample_choice != "--" and uploaded is None:
            uploaded = sample_choice

# ---------------------------
# Inference & display
# ---------------------------
if uploaded:
    if isinstance(uploaded, str):  # sample file path
        img = Image.open(uploaded).convert("RGB")
    else:
        img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")

    st.image(img, caption="Image input", use_column_width=True)

    # MobileNet inference
    if mobilenet is not None:
        X_m = image_to_mobilenet_input(img, target_size=(224,224))  # adapt if you used 128x128
        t0 = time.time()
        probs_m = mobilenet.predict(X_m)[0]
        t_m = (time.time() - t0)
        top5_idx_m = np.argsort(probs_m)[::-1][:5]
        top1_m = top5_idx_m[0]
    else:
        probs_m = None
        t_m = None
        top1_m = None
        top5_idx_m = []

    # DINOv2 inference (embedder + classifier)
    dino_time = None
    if dinov2_embedder is not None and dino_clf is not None:
        # detect device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            try:
                dinov2_embedder.to('cuda')
            except Exception:
                pass
        try:
            emb = image_to_dinov2_embedding(img, dinov2_embedder, device=device)
            t0 = time.time()
            probs_d = dino_clf.predict(emb)[0]
            dino_time = (time.time() - t0) + 0.0  # classifier time only
            top5_idx_d = np.argsort(probs_d)[::-1][:5]
            top1_d = top5_idx_d[0]
        except Exception as e:
            st.warning(f"Erreur inference DINOv2: {e}")
            probs_d = None
            top1_d = None
            top5_idx_d = []
    else:
        probs_d = None
        top1_d = None
        top5_idx_d = []

    # Load class names if provided
    class_names = None
    if os.path.exists("class_names.npy"):
        class_names = np.load("class_names.npy", allow_pickle=True)
    else:
        # fallback to indices only
        class_names = [str(i) for i in range(200)]

    # Display results
    st.subheader("Résultats")
    left, right = st.columns(2)

    with left:
        st.markdown("**MobileNetV2**")
        if probs_m is not None:
            st.markdown(f"- Top-1: **{class_names[top1_m]}** (prob={probs_m[top1_m]:.3f})")
            df_m = pd.DataFrame({
                "Classe": [class_names[i] for i in top5_idx_m],
                "Prob": [probs_m[i] for i in top5_idx_m]
            })
            st.table(df_m)
            st.write(f"Temps d'inférence (predict Keras): {t_m:.3f} s")
        else:
            st.write("MobileNet non chargé.")

    with right:
        st.markdown("**DINOv2 + Classifier**")
        if probs_d is not None:
            st.markdown(f"- Top-1: **{class_names[top1_d]}** (prob={probs_d[top1_d]:.3f})")
            df_d = pd.DataFrame({
                "Classe": [class_names[i] for i in top5_idx_d],
                "Prob": [probs_d[i] for i in top5_idx_d]
            })
            st.table(df_d)
            st.write(f"Temps d'inférence (classifier): {dino_time:.3f} s (embedding extraction sur CPU/GPU non comptée ici)")
        else:
            st.write("DINOv2 non chargé ou erreur.")

    # Summary table
    st.markdown("**Comparaison rapide**")
    summary = {
        "Modèle": ["MobileNetV2", "DINOv2+clf"],
        "Top-1 prédiction": [class_names[top1_m] if top1_m is not None else "—",
                             class_names[top1_d] if top1_d is not None else "—"],
        "Top-1 prob": [f"{probs_m[top1_m]:.3f}" if probs_m is not None else "—",
                       f"{probs_d[top1_d]:.3f}" if probs_d is not None else "—"],
        "Temps (s)": [f"{t_m:.3f}" if t_m is not None else "—",
                      f"{dino_time:.3f}" if dino_time is not None else "—"]
    }
    st.table(pd.DataFrame(summary))

st.markdown("---")
st.caption("Notes: DINOv2 embedder loaded via torch.hub may be large. For production, prefer precomputing embeddings or hosting the embedder on an inference service.")
