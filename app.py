import streamlit as st
from PIL import Image
import numpy as np
import os
import glob

###############################
# IMPORTS DES FONCTIONS
###############################

from mobilenet import load_mobilenet_model, predict_with_mobilenet
from dinov2 import load_dinov2_model, predict_with_dinov2

############################################
# CONFIG STREAMLIT (IMPORTANT POUR RENDER)
############################################
st.set_page_config(
    page_title="Stanford Dogs Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Stanford Dogs – MobileNetV2 & DINOv2")
st.write("Interface de prédiction pour MobileNetV2 et DINOv2.")

############################################
# CHARGEMENT DES MODÈLES
############################################

col1, col2 = st.columns(2)

with col1:
    st.header("Chargement des modèles")

    mobilenet_path = st.text_input(
        "Path MobileNetV2 (Keras model)",
        "best_mobilenetv2_finetuned.keras"
    )

    dino_classifier_path = st.text_input(
        "Path DINOv2 classifier (Keras)",
        "best_dinov2_classifier.keras"
    )

    load_button = st.button("Charger les modèles")

    mobilenet = None
    dinov2_embedder = None
    dino_clf = None

    # Chargement explicite par bouton
    if load_button:
        with st.spinner("Chargement des modèles..."):
            try:
                mobilenet = load_mobilenet_model(mobilenet_path)
                dinov2_embedder, dino_clf = load_dinov2_model(dino_classifier_path)
                st.success("Modèles chargés avec succès !")
            except Exception as e:
                st.error(f"Erreur lors du chargement : {e}")

    # Chargement automatique fallback
    else:
        try:
            mobilenet = load_mobilenet_model(mobilenet_path)
            dinov2_embedder, dino_clf = load_dinov2_model(dino_classifier_path)
        except:
            pass  # On ignore si les modèles ne sont pas présents

############################################
# UPLOAD UTILISATEUR ET EXEMPLES
############################################

with col2:
    st.header("Upload / Exemple")

    uploaded = st.file_uploader("Upload une image", type=["jpg", "jpeg", "png"])

    st.write("Ou tester une image du dataset :")

    sample_folder = os.path.join("images", "Images")
    sample_files = []

    if os.path.exists(sample_folder):
        sample_files = [
            os.path.join(sample_folder, d, f)
            for d in os.listdir(sample_folder)
            if os.path.isdir(os.path.join(sample_folder, d))
            for f in os.listdir(os.path.join(sample_folder, d))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    if sample_files:
        sample_choice = st.selectbox(
            "Images du dataset :", ["--"] + sample_files
        )

        if sample_choice != "--" and uploaded is None:
            uploaded = sample_choice

############################################
# PREDICTION SUR L’IMAGE UPLOADÉE
############################################

if uploaded:
    st.subheader("Image chargée")
    img = Image.open(uploaded).convert("RGB")
    st.image(img, use_column_width=True)

    st.subheader("Prédictions")

    if mobilenet:
        mobilenet_pred = predict_with_mobilenet(mobilenet, img)
        st.write("### MobileNetV2 :", mobilenet_pred)

    if dinov2_embedder and dino_clf:
        dinov2_pred = predict_with_dinov2(dinov2_embedder, dino_clf, img)
        st.write("### DINOv2 :", dinov2_pred)

############################################
# EXEMPLES AUTOMATIQUES : 5 premières classes
############################################

st.header("Exemples automatiques (5 premières classes)")

dataset_root = os.path.join("images", "Images")

classes = sorted([
    d for d in os.listdir(dataset_root)
    if os.path.isdir(os.path.join(dataset_root, d))
])

first_5 = classes[:5]
example_images = []

for cls in first_5:
    cls_folder = os.path.join(dataset_root, cls)
    imgs = sorted(glob.glob(os.path.join(cls_folder, "*")))

    imgs = [f for f in imgs if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if imgs:
        example_images.append((cls, imgs[0]))

if not example_images:
    st.warning("Aucune image trouvée dans les 5 premières classes.")
else:
    cols = st.columns(5)

    for idx, (cls, img_path) in enumerate(example_images):
        with cols[idx]:
            st.image(img_path, caption=cls, use_column_width=True)

            if st.button(f"Prédire ({cls})", key=f"auto_{idx}"):
                img = Image.open(img_path).convert("RGB")

                st.write(f"### Prédictions pour **{cls}** :")

                if mobilenet:
                    pred_m = predict_with_mobilenet(mobilenet, img)
                    st.write("MobileNetV2 :", pred_m)

                if dinov2_embedder and dino_clf:
                    pred_d = predict_with_dinov2(dinov2_embedder, dino_clf, img)
                    st.write("DINOv2 :", pred_d)

