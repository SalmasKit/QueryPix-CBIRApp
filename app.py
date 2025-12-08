# app.py

import streamlit as st
import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px

from cbir_logic import (
    hist_features,
    distance,
    build_index,
    search,
    create_histogram_plot
)

# ==========================
# CONFIGURATION DE LA PAGE
# ==========================
st.set_page_config(
    page_title="QueryPix - CBIR System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# INTERFACE STREAMLIT
# ==========================
def main():
    # CSS (inchangé)
    st.markdown("""<style>...</style>""", unsafe_allow_html=True)

    # --- En-tête
    st.markdown('<h1 class="main-header">🔍 QueryPix</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#666;">Content-Based Image Retrieval System</p>', unsafe_allow_html=True)

    # --- Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        color_space = st.selectbox("Espace couleur", ['RGB', 'HSV', 'Lab'])
        bins = st.select_slider("Nombre de bins", [8, 16, 32, 64, 256], 16)
        method = st.selectbox("Méthode de similarité", ['swain', 'euclidean', 'chi2', 'correlation'])

        st.markdown("---")
        st.subheader("📁 Dataset")

        dataset_folder = st.text_input("Chemin du dataset", "dataset/Corel-1k")

        if st.button("🧩 Construire l'index", use_container_width=True):
            if os.path.exists(dataset_folder):
                with st.spinner("Indexation en cours..."):
                    count = build_index(dataset_folder, bins, color_space)
                    st.success(f"Index construit avec {count} images")
            else:
                st.error("Dossier introuvable")

        st.markdown("---")
        st.subheader("🖼️ Image Requête")
        uploaded_file = st.file_uploader("Choisir une image", type=['jpg','jpeg','png'])

        if os.path.exists("descriptors.json"):
            with open("descriptors.json", "r") as f:
                db = json.load(f)
            st.info(f"Index : {len(db)} images")

    # --- Corps principal
    col1, col2 = st.columns([1, 2])

    # Image requête
    with col1:
        st.header("🎯 Image Requête")
        query_image = None

        if uploaded_file:
            query_image = Image.open(uploaded_file)
            st.image(query_image, caption="Image requête", use_container_width=True)
        else:
            st.info("📸 Veuillez uploader une image")

    # Analyse
    with col2:
        st.header("📊 Analyse")

        if query_image:
            st.subheader("📈 Histogrammes")
            fig = create_histogram_plot(query_image, color_space)
            st.plotly_chart(fig, use_container_width=True)

            if st.button("🔍 Lancer la recherche", type="primary", use_container_width=True):
                if os.path.exists("descriptors.json"):
                    results = search(query_image, "descriptors.json", bins, color_space, method)
                    st.session_state.results = results
                    st.session_state.query_image = query_image
                    st.session_state.dataset_folder = dataset_folder
                else:
                    st.error("Construisez l'index d'abord !")

    # Résultats
    if "results" in st.session_state:
        results = st.session_state.results
        dataset_folder = st.session_state.dataset_folder

        st.markdown("---")
        st.header("🎯 Résultats de la Recherche")

        colA, colB, colC, colD = st.columns(4)
        scores = [s for _, s in results]

        with colA: st.metric("Images trouvées", len(results))
        with colB: st.metric("Max", f"{max(scores):.3f}")
        with colC: st.metric("Min", f"{min(scores):.3f}")
        with colD: st.metric("Moyenne", f"{np.mean(scores):.3f}")

        fig_hist = px.histogram(scores, nbins=10, title="Distribution des scores")
        fig_hist.update_layout(template="plotly_dark")
        st.plotly_chart(fig_hist, use_container_width=True)

        # grille images
        st.subheader("🖼️ Images similaires")
        cols = st.columns(4)

        for i, (path, score) in enumerate(results):
            with cols[i % 4]:
                full = os.path.join(dataset_folder, path)
                if os.path.exists(full):
                    img = Image.open(full)
                    st.image(img, caption=f"{os.path.basename(path)}\nScore: {score:.3f}", use_container_width=True)

        # tableau
        df = pd.DataFrame(results, columns=["Image", "Score"])
        df["Image"] = df["Image"].apply(os.path.basename)
        df = df.sort_values("Score", ascending=False)

        st.dataframe(df, use_container_width=True)

    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#666;'>🔍 QueryPix CBIR • 2025/2026</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
