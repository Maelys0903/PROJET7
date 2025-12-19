import streamlit as st
from PIL import Image
import numpy as np
import os
import glob
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

st.title("Stanford Dogs – MobileNetV2 & DINOv2")
st.write("Interface de prédiction pour MobileNetV2 et DINOv2.")

############################################
# CHARGEMENT DES MODÈLES (AUTO)
############################################
@st.cache_resource
def load_models():
    mobilenet = load_model("best_mobilenetv2_finetuned.keras")
    dino = load_model("best_dinov2_classifier.keras")
    return mobilenet, dino

try:
    mobilenet, dino_clf = load_models()
    st.success("✅ Modèles chargés avec succès")
except Exception as e:
    st.error("❌ Impossible de charger les modèles")
    st.exception(e)
    mobilenet, dino_clf = None, None

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
        sample_choice = st.selectbox(
            "Images du dataset :", ["--"] + sample_files
        )
        if sample_choice != "--" and uploaded is None:
            uploaded = sample_choice

############################################
# FONCTION DE PRÉDICTION
############################################
def predict_with_model(model, img):
    # Taille attendue par le modèle
    input_shape = model.input_shape
    print(input_shape)
    target_size = (input_shape[1], input_shape[2])

    img_resized = img.resize(target_size)
    x = image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    preds = model.predict(x, verbose=0)
    class_idx = int(np.argmax(preds, axis=1)[0])
    class_name = classes[class_idx] if class_idx < len(classes) else f"Classe {class_idx}"

    return class_name, float(preds[0][class_idx])

############################################
# PRÉDICTION IMAGE UTILISATEUR
############################################
with col2:
    if uploaded and mobilenet and dino_clf:
        img = Image.open(uploaded).convert("RGB")
        st.subheader("Image analysée")
        st.image(img, width=350)

        st.subheader("Prédictions")
        class_name, prob = predict_with_model(mobilenet, img)
        st.write(f"### MobileNetV2 : {class_name} ({prob:.2f})")

        class_name, prob = predict_with_model(dino_clf, img)
        st.write(f"### DINOv2 : {class_name} ({prob:.2f})")

############################################
# EXEMPLES AUTOMATIQUES
############################################
st.header("Exemples automatiques (5 premières classes)")

if classes and mobilenet and dino_clf:
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
        for idx, (cls, img_path) in enumerate(example_images):
            with cols[idx]:
                img = Image.open(img_path).convert("RGB")
                st.image(img, caption=cls, width=200)

                cname, prob = predict_with_model(mobilenet, img)
                st.caption(f"MobileNetV2 : {cname} ({prob:.2f})")

                cname, prob = predict_with_model(dino_clf, img)
                st.caption(f"DINOv2 : {cname} ({prob:.2f})")

