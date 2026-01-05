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
import base64

# ============================================================
# RACCOURCI NOMS CLASSES
# ============================================================

def clean_class_name(class_name: str) -> str:
    if "-" in class_name:
        return class_name.split("-", 1)[1]
    return class_name

# ============================================================
# CONFIGURATION STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Stanford Dogs – Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    /* Supprime l’espace blanc en haut */
    .block-container {
        padding-top: 1rem;
    }

    /* Réduit la hauteur du header Streamlit */
    header[data-testid="stHeader"] {
        height: 0.5rem;
    }

    /* Supprime le fond gris du header */
    header[data-testid="stHeader"] {
        background: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .image-card {
        height: 240px;
        width: 100%;
        overflow: hidden;
        border-radius: 10px;
        background-color: #f2f2f2;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 0.5rem;
    }

    .image-card img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Stanford Dogs – Dashboard d’analyse des modèles")
st.write(
    """
    Ce dashboard présente une **analyse comparative avancée**
    entre deux modèles de classification d’images :
    **MobileNetV2** et **DINOv2**.

    Les prédictions sont **pré-calculées** afin de garantir
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

st.sidebar.title("Filtres")

class_choice = st.sidebar.selectbox(
    "Classe réelle",
    ["Aléatoire"] + sorted(df["true_class"].unique()),
    format_func=lambda x: "Toutes" if x == "Toutes" else clean_class_name(x)
)

n_images = st.sidebar.slider(
    "Nombre d’images affichées",
    min_value=3,
    max_value=12,
    value=6,
    step=3,
    key="n_images_slider"
)

# ============================================================
# APPLICATION DES FILTRES
# ============================================================

df_view = df.copy()

if class_choice != "Aléatoire":
    df_view = df_view[df_view["true_class"] == class_choice]

st.write(f"**{len(df_view)} images** sélectionnées")

if len(df_view) == 0:
    st.warning("Aucune image disponible avec ces filtres.")
    st.stop()

# ============================================================
# GALERIE D’IMAGES
# ============================================================

st.subheader("Galerie d’images avec prédictions")

sample_df = df_view.sample(min(n_images, len(df_view)))
cols = st.columns(3)

for i, (_, row) in enumerate(sample_df.iterrows()):
    with cols[i % 3]:

        # ----------------------------------------------------
        # CHEMIN IMAGE (VERSION QUI FONCTIONNE SUR RENDER)
        # ----------------------------------------------------
        csv_path = row["image_path"].replace("\\", "/")
        image_path = os.path.normpath(os.path.join(BASE_DIR, csv_path))

        if not os.path.exists(image_path):
            st.error("❌ Image introuvable")
            continue

        # ----------------------------------------------------
        # IMAGE (HAUTEUR FIXE – ALIGNEMENT PARFAIT)
        # ----------------------------------------------------
        with open(image_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode()

        st.markdown(
            f"""
            <div class="image-card">
                <img src="data:image/jpeg;base64,{img_base64}">
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption(f"Classe réelle : {clean_class_name(row['true_class'])}")

        # ----------------------------------------------------
        # ÉVALUATION DES MODÈLES
        # ----------------------------------------------------
        mn_correct = row["mobilenet_pred"] == row["true_class"]
        dn_correct = row["dinov2_pred"] == row["true_class"]

        st.markdown(
            f"""
            🟦 **MobileNetV2**  
            `{clean_class_name(row['mobilenet_pred'])}`  
            Proba : **{row['mobilenet_proba']:.2f}**  
            {'✅ Correct' if mn_correct else '❌ Incorrect'}
            ---
            🟩 **DINOv2**  
            `{clean_class_name(row['dinov2_pred'])}`  
            Proba : **{row['dinov2_proba']:.2f}**  
            {'✅ Correct' if dn_correct else '❌ Incorrect'}
            """
        )

# ============================================================
# ANALYSE GLOBALE DES PERFORMANCES (IMAGES AFFICHÉES)
# ============================================================

st.divider()
st.subheader("Analyse globale des performances (images affichées)")

# Sécurité si peu d’images
if len(sample_df) == 0:
    st.warning("Aucune image sélectionnée pour l’analyse.")
else:
    acc_mn = (sample_df["mobilenet_pred"] == sample_df["true_class"]).mean()
    acc_dn = (sample_df["dinov2_pred"] == sample_df["true_class"]).mean()

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
        text=acc_df["Accuracy"].map(lambda x: f"{x:.0%}"),
        title="Accuracy sur les images affichées"
    )

    fig_acc.update_layout(
        yaxis=dict(range=[0, 1]),
        font=dict(size=14)
    )

    st.plotly_chart(fig_acc, use_container_width=True)

# ============================================================
# ANALYSE PAR CLASSE (GRAPHIQUE INTERACTIF)
# ============================================================

st.subheader("Comparaison des performances par classe")

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
