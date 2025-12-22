import streamlit as st
from PIL import Image
import numpy as np
import os
import glob

# TensorFlow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# PyTorch (DINOv2 backbone)
import torch
import torchvision.transforms as T

############################################
# CONFIG STREAMLIT
############################################
st.set_page_config(
    page_title="Stanford Dogs Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Stanford Dogs – MobileNetV2 & DINOv2")
st.write("Interface de prédiction pour MobileNetV2 et DINOv2.")

############################################
# CHARGEMENT DES MODÈLES LÉGERS (BOOT)
############################################
@st.cache_resource
def load_light_models():
    mobilenet = load_model("best_mobilenetv2_finetuned.keras")
    dino_classifier = load_model("best_dinov2_classifier.keras")
    return mobilenet, dino_classifier

try:
    mobilenet, dino_clf = load_light_models()
    st.success("✅ Modèles Keras chargés")
except Exception as e:
    st.error("❌ Impossible de charger les modèles")
    st.exception(e)
    st.stop()

############################################
# CHARGEMENT DINOv2 BACKBONE (À LA DEMANDE)
############################################
@st.cache_resource
def load_dino_backbone():
    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vitb14"
    )
    model.eval()

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return model, transform

############################################
# CHARGEMENT DES CLASSES
############################################
dataset_root = os.path.join("images", "Images")

if os.path.exists(dataset_root):
    classes = sorted([
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    ])
else:
    classes = []

if not classes:
    st.warning("⚠️ Aucune classe trouvée dans images/Images")

############################################
# LAYOUT
############################################
col1, col2 = st.columns(2)

############################################
# UPLOAD UTILISATEUR
############################################
with col1:
    st.header("Upload / Exemple")
    uploaded = st.file_uploader("Upload une image", type=["jpg", "jpeg", "png"])

    st.write("Ou tester une image du dataset :")

    sample_files = []
    if os.path.exists(dataset_root):
        sample_files = [
            os.path.join(dataset_root, d, f)
            for d in os.listdir(dataset_root)
            if os.path.isdir(os.path.join(dataset_root, d))
            for f in os.listdir(os.path.join(dataset_root, d))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    if sample_files:
        sample_choice = st.selectbox("Images du dataset :", ["--"] + sample_files)
        if sample_choice != "--" and uploaded is None:
            uploaded = sample_choice

############################################
# PRÉDICTION MOBILENET
############################################
def predict_mobilenet(model, img):
    target_size = (model.input_shape[1], model.input_shape[2])
    img_resized = img.resize(target_size)

    x = image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0) / 255.0

    preds = model.predict(x, verbose=0)
    idx = int(np.argmax(preds, axis=1)[0])
    return classes[idx], float(preds[0][idx])

############################################
# PRÉDICTION DINOv2 (PIPELINE CORRECT)
############################################
def predict_dinov2(img, classifier):
    with st.spinner("Chargement DINOv2 (première utilisation)..."):
        backbone, transform = load_dino_backbone()

    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        embedding = backbone(img_t).cpu().numpy()  # (1, 768)

    preds = classifier.predict(embedding, verbose=0)
    idx = int(np.argmax(preds, axis=1)[0])
    return classes[idx], float(preds[0][idx])

############################################
# IMAGE UTILISATEUR
############################################
with col2:
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.subheader("Image analysée")
        st.image(img, width=350)

        st.subheader("Prédictions")

        cname, prob = predict_mobilenet(mobilenet, img)
        st.write(f"### MobileNetV2 : {cname} ({prob:.2f})")

        cname, prob = predict_dinov2(img, dino_clf)
        st.write(f"### DINOv2 : {cname} ({prob:.2f})")

############################################
# EXEMPLES AUTOMATIQUES
############################################
st.header("Exemples automatiques (5 premières classes)")

if classes:
    first_5 = classes[:5]
    example_images = []

    for cls in first_5:
        cls_folder = os.path.join(dataset_root, cls)
        imgs = glob.glob(os.path.join(cls_folder, "*"))
        imgs = [f for f in imgs if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if imgs:
            example_images.append((cls, imgs[0]))

    if example_images:
        cols = st.columns(len(example_images))
        for i, (cls, path) in enumerate(example_images):
            with cols[i]:
                img = Image.open(path).convert("RGB")
                st.image(img, caption=cls, width=200)

                cname, prob = predict_mobilenet(mobilenet, img)
                st.caption(f"MobileNetV2 : {cname} ({prob:.2f})")

                cname, prob = predict_dinov2(img, dino_clf)
                st.caption(f"DINOv2 : {cname} ({prob:.2f})")
