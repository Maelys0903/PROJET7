import streamlit as st
from PIL import Image
import numpy as np
import os
import glob

# TensorFlow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# PyTorch / DINOv2
import torch
import joblib
from torchvision import transforms

############################################
# SÉCURITÉ RENDER FREE
############################################
torch.set_num_threads(1)

############################################
# CONFIG STREAMLIT
############################################
st.set_page_config(
    page_title="Stanford Dogs Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Stanford Dogs – MobileNetV2 & DINOv2")
st.write("Interface de prédiction d'images de chiens")

############################################
# CHARGEMENT DES CLASSES
############################################
dataset_root = os.path.join("images", "Images")

if os.path.exists(dataset_root):
    classes = sorted(
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    )
else:
    classes = []

if not classes:
    st.warning("Aucune classe trouvée dans images/Images")

############################################
# CHARGEMENT DU MODÈLE MOBILENET (BOOT UNIQUE)
############################################
if "mobilenet" not in st.session_state:
    with st.spinner("Chargement du modèle MobileNetV2..."):
        try:
            st.session_state.mobilenet = load_model(
                "best_mobilenetv2_finetuned.keras",
                compile=False
            )
            st.success("Modèle MobileNetV2 chargé")
        except Exception as e:
            st.error("Impossible de charger MobileNetV2")
            st.exception(e)
            st.stop()

mobilenet = st.session_state.mobilenet

############################################
# CHARGEMENT DU MODÈLE DINOv2 (BOOT UNIQUE)
############################################
if "dinov2" not in st.session_state:
    with st.spinner("Chargement du modèle DINOv2..."):
        try:
            st.session_state.dinov2_backbone = torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vits14",
                pretrained=True
            )
            st.session_state.dinov2_backbone.eval()

            st.session_state.dinov2_clf = joblib.load(
                "dinov2_classifier.pkl"
            )

            st.success("Modèle DINOv2 chargé")
        except Exception as e:
            st.error("Impossible de charger DINOv2")
            st.exception(e)
            st.stop()

dinov2_backbone = st.session_state.dinov2_backbone
dinov2_clf = st.session_state.dinov2_clf

############################################
# TRANSFORM DINOv2
############################################
dinov2_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

############################################
# FONCTIONS DE PRÉDICTION
############################################
def predict_mobilenet(model, img):
    target_size = (model.input_shape[1], model.input_shape[2])
    img_resized = img.resize(target_size)

    x = image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0) / 255.0

    preds = model.predict(x, verbose=0)
    idx = int(np.argmax(preds, axis=1)[0])

    return classes[idx], float(preds[0][idx])


def predict_dinov2(backbone, clf, img):
    x = dinov2_transform(img).unsqueeze(0)

    with torch.no_grad():
        feats = backbone(x)
        feats = feats.cpu().numpy()

    probs = clf.predict_proba(feats)[0]
    idx = int(np.argmax(probs))

    return classes[idx], float(probs[idx])

############################################
# LAYOUT
############################################
col1, col2 = st.columns(2)

############################################
# COLONNE GAUCHE – INPUT
############################################
with col1:
    st.header("Upload / Exemple")

    uploaded = st.file_uploader(
        "Upload une image",
        type=["jpg", "jpeg", "png"]
    )

    st.subheader("Choix du modèle")
    model_choice = st.radio(
        "Modèle de prédiction :",
        ["MobileNetV2", "DINOv2"]
    )

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
        sample_choice = st.selectbox(
            "Images du dataset :",
            ["--"] + sample_files
        )
        if sample_choice != "--" and uploaded is None:
            uploaded = sample_choice

############################################
# COLONNE DROITE – PRÉDICTION
############################################
with col2:
    if uploaded:
        img = Image.open(uploaded).convert("RGB")

        st.subheader("Image analysée")
        st.image(img, width=350)

        st.subheader("Prédiction")

        if model_choice == "MobileNetV2":
            cname, prob = predict_mobilenet(mobilenet, img)
            st.write(f"###MobileNetV2 → {cname} ({prob:.2f})")
        else:
            cname, prob = predict_dinov2(dinov2_backbone, dinov2_clf, img)
            st.write(f"###DINOv2 → {cname} ({prob:.2f})")

############################################
# EXEMPLES AUTOMATIQUES
############################################
st.header("Exemples automatiques (5 premières classes)")

if classes:
    first_5 = classes[:5]
    example_images = []

    for cls in first_5:
        cls_folder = os.path.join(dataset_root, cls)
        imgs = [
            f for f in glob.glob(os.path.join(cls_folder, "*"))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if imgs:
            example_images.append((cls, imgs[0]))

    if example_images:
        cols = st.columns(len(example_images))
        for i, (cls, path) in enumerate(example_images):
            with cols[i]:
                img = Image.open(path).convert("RGB")
                st.image(img, caption=cls, width=200)

                if model_choice == "MobileNetV2":
                    cname, prob = predict_mobilenet(mobilenet, img)
                else:
                    cname, prob = predict_dinov2(dinov2_backbone, dinov2_clf, img)

                st.caption(f"{cname} ({prob:.2f})")
