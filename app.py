# ============================================================
# Dashboard Streamlit – Stanford Dogs
# Analyse et comparaison MobileNetV2 vs DINOv2
# Prédictions pré-calculées (mode production léger)
# ============================================================

import streamlit as st
import pandas as pd
from PIL import Image
import os
import plotly.express as px

# ============================================================
# CONFIGURATION STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Stanford Dogs – Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🐶 Stanford Dogs – Dashboard d’analyse des modèles")
st.write(
    """
    Ce dashboard présente une **analyse comparative avancée**
    entre deux modèles de classification d’images :
    **MobileNetV2** et **DINOv2 (ViT-B/14)**.

    👉 Les prédictions sont **pré-calculées** afin de garantir
    des performances optimales et une compatibilité avec Render (free).
    """
)

# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

@st.cache_data
def load_predictions():
    return pd.read_csv("predictions_mobilenet_dinov2.csv")

df = load_predictions()

BASE_DIR = os.path.dirname(__file__)

# ============================================================
# SIDEBAR – FILTRES
# ============================================================

st.sidebar.title("🎛️ Filtres")

class_choice = st.sidebar.selectbox(
    "Classe réelle",
    ["Toutes"] + sorted(df["true_class"].unique())
)

n_images = st.sidebar.slider(
    "Nombre d’images affichées",
    min_value=3,
    max_value=12,
    value=6,
    step=3
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
# GALERIE D’IMAGES
# ============================================================

st.subheader("Galerie d’images avec prédictions")

N_IMAGES = 6  # nombre d’images affichées
sample_df = df_view.sample(min(N_IMAGES, len(df_view)))

cols = st.columns(3)

for i, (_, row) in enumerate(sample_df.iterrows()):
    with cols[i % 3]:

        # Reconstruction du chemin image
        relative_path = row["image_path"].split("Images")[-1]
        relative_path = relative_path.lstrip("/\\")
        image_path = os.path.join(BASE_DIR, "images", "Images", relative_path)

        if os.path.exists(image_path):
            img = Image.open(image_path).convert("RGB")
            st.image(img, width=250)

            st.markdown(
                f"""
                **Classe réelle :** {row['true_class']}

                🟦 **MobileNetV2**  
                → {row['mobilenet_pred']}  
                *(proba : {row['mobilenet_proba']:.2f})*

                🟩 **DINOv2**  
                → {row['dinov2_pred']}  
                *(proba : {row['dinov2_proba']:.2f})*
                """
            )
        else:
            st.error("Image introuvable")

# ============================================================
# ANALYSE GLOBALE DES PERFORMANCES
# ============================================================

st.divider()
st.subheader("📈 Analyse globale des performances")

acc_mn = (df["mobilenet_pred"] == df["true_class"]).mean()
acc_dn = (df["dinov2_pred"] == df["true_class"]).mean()

acc_df = pd.DataFrame({
    "Modèle": ["MobileNetV2", "DINOv2"],
    "Accuracy": [acc_mn, acc_dn]
})

fig_acc = px.bar(
    acc_df,
    x="Modèle",
    y="Accuracy",
    color="Modèle",
    color_discrete_map={
        "MobileNetV2": "#1f77b4",  # bleu foncé (WCAG)
        "DINOv2": "#2ca02c"        # vert foncé (WCAG)
    },
    title="Accuracy globale par modèle"
)

fig_acc.update_layout(
    yaxis_tickformat=".0%",
    font=dict(size=14)
)

st.plotly_chart(fig_acc, use_container_width=True)

# ============================================================
# ANALYSE PAR CLASSE (GRAPHIQUE INTERACTIF)
# ============================================================

st.subheader("📊 Comparaison des performances par classe")

acc_per_class = (
    df
    .assign(
        mn_correct=lambda x: x["mobilenet_pred"] == x["true_class"],
        dn_correct=lambda x: x["dinov2_pred"] == x["true_class"]
    )
    .groupby("true_class")[["mn_correct", "dn_correct"]]
    .mean()
    .reset_index()
)

acc_per_class.columns = ["Classe", "MobileNetV2", "DINOv2"]

fig_class = px.scatter(
    acc_per_class,
    x="MobileNetV2",
    y="DINOv2",
    hover_name="Classe",
    labels={
        "MobileNetV2": "Accuracy MobileNetV2",
        "DINOv2": "Accuracy DINOv2"
    },
    title="Comparaison des performances par classe"
)

fig_class.update_traces(marker=dict(size=10))
fig_class.update_layout(font=dict(size=14))

st.plotly_chart(fig_class, use_container_width=True)

# ============================================================
# ACCESSIBILITÉ (WCAG)
# ============================================================

st.info(
    "♿ **Accessibilité (WCAG)** : "
    "couleurs contrastées, tailles de police lisibles, "
    "informations redondantes (texte + couleur) "
    "et graphiques interactifs accessibles."
)

# ============================================================
# PIED DE PAGE
# ============================================================

st.divider()
st.caption(
    "Projet Computer Vision – Stanford Dogs | "
    "Dashboard Streamlit – Comparaison MobileNetV2 vs DINOv2"
)
