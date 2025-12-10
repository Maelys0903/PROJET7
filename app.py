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
    dino_clf = None

    if load_button:
        with st.spinner("Chargement des modèles..."):
            try:
                mobilenet = load_model(mobilenet_path)
                dino_clf = load_model(dino_classifier_path)
                st.success("Modèles chargés avec succès !")
            except Exception as e:
                st.error(f"Erreur lors du chargement : {e}")

############################################
# CHARGEMENT DES CLASSES
############################################
dataset_root = os.path.join("images", "Images")
classes = sorted([
    d for d in os.listdir(dataset_root)
    if os.path.isdir(os.path.join(dataset_root, d))
])
if not classes:
    st.error("Aucune classe trouvée dans le dossier images/Images.")

############################################
# UPLOAD UTILISATEUR ET EXEMPLES
############################################

with col2:
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
# FONCTION DE PREDICTION
############################################
def predict_with_model(model, img, target_size=(224, 224)):
    img_resized = img.resize(target_size)
    x = image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Normalisation
    preds = model.predict(x)
    class_idx = np.argmax(preds, axis=1)[0]
    class_name = classes[class_idx] if class_idx < len(classes) else f"Classe {class_idx}"
    return class_name, preds[0][class_idx]

############################################
# PREDICTION SUR L’IMAGE UPLOADÉE
############################################
if uploaded:
    st.subheader("Image chargée")
    img = Image.open(uploaded).convert("RGB")
    st.image(img, width=400)

    st.subheader("Prédictions")

    if mobilenet:
        class_name, prob = predict_with_model(mobilenet, img)
        st.write(f"### MobileNetV2 : {class_name}, Probabilité {prob:.2f}")

    if dino_clf:
        class_name, prob = predict_with_model(dino_clf, img)
        st.write(f"### DINOv2 : {class_name}, Probabilité {prob:.2f}")

############################################
# EXEMPLES AUTOMATIQUES : 5 premières classes
############################################
st.header("Exemples automatiques (5 premières classes)")

first_5 = classes[:5]
example_images = []

for cls in first_5:
    cls_folder = os.path.join(dataset_root, cls)
    imgs = sorted(glob.glob(os.path.join(cls_folder, "*")))
    imgs = [f for f in imgs if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if imgs:
        example_images.append((cls, imgs[0]))

if example_images:
    cols = st.columns(5)
    for idx, (cls, img_path) in enumerate(example_images):
        with cols[idx]:
            st.image(img_path, caption=cls, width=200)
            if st.button(f"Prédire ({cls})", key=f"auto_{idx}"):
                img = Image.open(img_path).convert("RGB")
                st.write(f"### Prédictions pour **{cls}** :")
                if mobilenet:
                    class_name, prob = predict_with_model(mobilenet, img)
                    st.write(f"MobileNetV2 : {class_name}, Probabilité {prob:.2f}")
                if dino_clf:
                    class_name, prob = predict_with_model(dino_clf, img)
                    st.write(f"DINOv2 : {class_name}, Probabilité {prob:.2f}")

