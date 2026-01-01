# ============================================================
# Dashboard Streamlit – Stanford Dogs
# Comparaison MobileNetV2 vs DINOv2 (prédictions pré-calculées)
# ============================================================

import streamlit as st
import pandas as pd
from PIL import Image
import os

# ============================================================
# CONFIGURATION STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Stanford Dogs – Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🐶 Stanford Dogs – Dashboard de prédictions")
st.write(
    """
    Ce dashboard compare les prédictions de deux modèles :
    **MobileNetV2** et **DINOv2 (ViT-B/14)**  
    Les prédictions sont **pré-calculées** afin de garantir
    un affichage rapide et compatible avec Render (free).
    """
)

# ============================================================
# CHARGEMENT DES DONNÉES (CSV)
# ============================================================

@st.cache_data
def load_predictions():
    """Charge le fichier CSV contenant toutes les prédictions."""
    return pd.read_csv("predictions_mobilenet_dinov2.csv")

df = load_predictions()

# ============================================================
# SIDEBAR – FILTRES UTILISATEUR
# ============================================================

st.sidebar.title("🎛️ Filtres")

class_choice = st.sidebar.selectbox(
    "Classe réelle",
    ["Toutes"] + sorted(df["true_class"].unique())
)

# ============================================================
# APPLICATION DES FILTRES
# ============================================================

df_view = df.copy()

if class_choice != "Toutes":
    df_view = df_view[df_view["true_class"] == class_choice]

st.write(f"📊 **{len(df_view)} images** sélectionnées")

if len(df_view) == 0:
    st.warning("Aucune image disponible avec ces filtres.")
    st.stop()

# ============================================================
# SÉLECTION D'UNE IMAGE
# ============================================================

row = df_view.sample(1).iloc[0]

# ============================================================
# CORRECTION CHEMIN IMAGE (IMPORTANT POUR RENDER)
# ============================================================

# Dossier racine du projet (app.py)
BASE_DIR = os.path.dirname(__file__)

# Chemin issu du CSV (peut contenir des "\" Windows)
csv_path = row["image_path"]

# Normalisation : remplace "\" par "/" puis normalise le chemin
csv_path = csv_path.replace("\\", "/")

# Construction du chemin absolu utilisé par Streamlit
image_path = os.path.normpath(os.path.join(BASE_DIR, csv_path))

# ============================================================
# AFFICHAGE IMAGE
# ============================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Image analysée")

    if os.path.exists(image_path):
        img = Image.open(image_path).convert("RGB")
        st.image(img, width=350)
        st.caption(f"Classe réelle : {row['true_class']}")
    else:
        st.error("❌ Image introuvable")
        st.write("Chemin cherché :", image_path)

# ============================================================
# AFFICHAGE DES PRÉDICTIONS
# ============================================================

with col2:
    st.subheader("🧠 Prédictions")

    st.markdown(
        f"""
        ### 🔵 MobileNetV2  
        **Classe prédite :** {row['mobilenet_pred']}  
        **Probabilité :** {row['mobilenet_proba']:.2f}

        ### 🟢 DINOv2 (ViT-B/14)  
        **Classe prédite :** {row['dinov2_pred']}  
        **Probabilité :** {row['dinov2_proba']:.2f}
        """
    )

# ============================================================
# ANALYSE RAPIDE
# ============================================================

st.divider()
st.subheader("✅ Analyse rapide")

mn_correct = row["mobilenet_pred"] == row["true_class"]
dn_correct = row["dinov2_pred"] == row["true_class"]

st.write(
    f"""
    - MobileNetV2 : {'✅ Correct' if mn_correct else '❌ Incorrect'}
    - DINOv2 : {'✅ Correct' if dn_correct else '❌ Incorrect'}
    """
)

# ============================================================
# PIED DE PAGE
# ============================================================

st.divider()
st.caption(
    "Projet Computer Vision – Stanford Dogs | "
    "Dashboard Streamlit – Prédictions pré-calculées"
)
