import streamlit as st
from PIL import Image
import numpy as np
import os
import glob

# TensorFlow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

############################################
# CONFIG STREAMLIT
############################################
st.set_page_config(
    page_title="Stanford Dogs Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Stanford Dogs – MobileNetV2")
st.write("Interface de prédiction basée sur MobileNetV2")

############################################
# CHARGEMENT DU MODÈLE (BOOT UNIQUE)
############################################
if "mobilenet" not in st.session_state:
    with st.spinner("Chargement du modèle MobileNetV2..."):
        try:
            st.session_state.mobilenet = load_model(
                "best_mobilenetv2_finetuned.keras",
                compile=False
            )
            st.success("✅ Modèle MobileNetV2 chargé")
        except Exception as e:
            st.error("❌ Impossible de charger le modèle")
            st.exception(e)
            st.stop()

mobilenet = st.session_state.mobilenet

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

    uploaded = st.file_uploader(
        "Upload une image",
        type=["jpg", "jpeg", "png"]
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
# IMAGE UTILISATEUR & PRÉDICTION
############################################
with col2:
    if uploaded:
        img = Image.open(uploaded).convert("RGB")

        st.subheader("Image analysée")
        st.image(img, width=350)

        st.subheader("Prédiction")

        cname, prob = predict_mobilenet(mobilenet, img)
        st.write(f"### 🐶 {cname} ({prob:.2f})")

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

                cname, prob = predict_mobilenet(mobilenet, img)
                st.caption(f"{cname} ({prob:.2f})")

