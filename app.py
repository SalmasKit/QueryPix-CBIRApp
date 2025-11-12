"""
CBIR Modern - Corel-1000
Auteur : Salma Barrak
Année : 2025/2026
Application Streamlit moderne pour la recherche d'images par le contenu
"""

import streamlit as st
import os
import cv2
import json
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import time

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
# FONCTIONS CBIR
# ==========================
def hist_features(image, bins=16, color_space='RGB'):
    """Calcule un histogramme normalisé (3 canaux)."""
    if color_space == 'HSV':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif color_space == 'Lab':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    features = []
    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    return np.array(features)

def distance(HQ, HI, method='swain'):
    """Calcule la distance entre deux histogrammes."""
    if method == 'swain':  # Histogram Intersection
        return np.sum(np.minimum(HQ, HI)) / np.sum(HQ)
    elif method == 'euclidean':
        return -np.linalg.norm(HQ - HI)
    elif method == 'chi2':
        eps = 1e-10
        return -0.5 * np.sum(((HQ - HI) ** 2) / (HQ + HI + eps))
    elif method == 'correlation':
        HQ_mean = HQ - np.mean(HQ)
        HI_mean = HI - np.mean(HI)
        corr = np.sum(HQ_mean * HI_mean) / (
            np.sqrt(np.sum(HQ_mean ** 2)) * np.sqrt(np.sum(HI_mean ** 2)) + 1e-10)
        return corr
    else:
        raise ValueError("Méthode inconnue")

def build_index(dataset_folder, bins=16, color_space='RGB'):
    """Construit et sauvegarde les descripteurs du dataset."""
    descriptors = {}
    count = 0
    
    # Compter le nombre total d'images
    total_images = 0
    for root, dirs, files in os.walk(dataset_folder):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                total_images += 1
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for root, dirs, files in os.walk(dataset_folder):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(root, filename)
                image = cv2.imread(path)
                if image is None:
                    continue
                image = cv2.resize(image, (256, 256))
                features = hist_features(image, bins, color_space)
                rel_path = os.path.relpath(path, dataset_folder)
                descriptors[rel_path] = features.tolist()
                count += 1
                
                # Mise à jour de la progression
                progress = (count / total_images) * 100
                progress_bar.progress(progress / 100)
                status_text.text(f"Indexation: {count}/{total_images} images...")
    
    with open("descriptors.json", "w") as f:
        json.dump(descriptors, f, indent=2)
    
    progress_bar.empty()
    status_text.empty()
    return count

def search(query_image, descriptors_file, bins=16, color_space='RGB', method='swain'):
    """Recherche les images les plus similaires."""
    if not os.path.exists(descriptors_file):
        st.error("Le fichier descriptors.json n'existe pas.")
        return []

    with open(descriptors_file, "r") as f:
        db = json.load(f)

    # Convertir l'image PIL en array numpy pour OpenCV
    query_img = np.array(query_image)
    query_img = cv2.cvtColor(query_img, cv2.COLOR_RGB2BGR)
    query_img = cv2.resize(query_img, (256, 256))
    query_features = hist_features(query_img, bins, color_space)

    results = []
    total = len(db)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (rel_path, desc) in enumerate(db.items()):
        desc_array = np.array(desc)
        score = distance(query_features, desc_array, method)
        results.append((rel_path, score))
        
        # Mise à jour de la progression
        if i % 50 == 0:
            progress = (i / total) * 100
            progress_bar.progress(progress / 100)
            status_text.text(f"Recherche: {i}/{total} images...")

    results.sort(key=lambda x: x[1], reverse=True)
    
    progress_bar.empty()
    status_text.empty()
    return results[:12]

def create_histogram_plot(query_image, color_space='RGB'):
    """Crée un graphique d'histogramme interactif avec Plotly."""
    # Convertir l'image PIL en array numpy pour OpenCV
    img_array = np.array(query_image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_array = cv2.resize(img_array, (256, 256))
    
    # Calculer les histogrammes
    hist_full = hist_features(img_array, bins=256, color_space=color_space)
    hist_bin = hist_features(img_array, bins=16, color_space=color_space)
    
    # Créer les subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Histogramme Complet (256 bins)', 'Histobine (16 bins)'),
        horizontal_spacing=0.1
    )
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'] if color_space == 'RGB' else ['#FF9FF3', '#54A0FF', '#FF9F43']
    color_names = ['Rouge', 'Vert', 'Bleu'] if color_space == 'RGB' else ['Canal 1', 'Canal 2', 'Canal 3']
    
    # Histogramme complet
    for i in range(3):
        fig.add_trace(
            go.Scatter(
                x=list(range(256)),
                y=hist_full[i*256:(i+1)*256],
                mode='lines',
                name=f'{color_names[i]} - Complet',
                line=dict(color=colors[i], width=2),
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Histobine
    for i in range(3):
        fig.add_trace(
            go.Scatter(
                x=list(range(16)),
                y=hist_bin[i*16:(i+1)*16],
                mode='lines+markers',
                name=f'{color_names[i]} - Bins',
                line=dict(color=colors[i], width=2),
                marker=dict(size=6),
                showlegend=True
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        height=400,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.update_xaxes(title_text='Valeur de pixel', row=1, col=1)
    fig.update_xaxes(title_text='Bin', row=1, col=2)
    fig.update_yaxes(title_text='Fréquence normalisée', row=1, col=1)
    fig.update_yaxes(title_text='Fréquence normalisée', row=1, col=2)
    
    return fig

# ==========================
# INTERFACE STREAMLIT
# ==========================
def main():
    # CSS personnalisé pour un look moderne
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 2rem;
        }
        .card {
            background: rgba(255,255,255,0.1);
            padding: 1.5rem;
            border-radius: 15px;
            border: 1px solid rgba(255,255,255,0.2);
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # En-tête principal
    st.markdown('<h1 class="main-header">🔍 QueryPix</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 3rem;">Content-Based Image Retrieval System</p>', unsafe_allow_html=True)
    
    # Sidebar pour la configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Paramètres de recherche
        color_space = st.selectbox(
            "Espace couleur",
            ['RGB', 'HSV', 'Lab'],
            index=0,
            help="Espace colorimétrique pour l'analyse"
        )
        
        bins = st.select_slider(
            "Nombre de bins",
            options=[8, 16, 32, 64, 256],
            value=16,
            help="Nombre de bins pour l'histogramme"
        )
        
        method = st.selectbox(
            "Méthode de similarité",
            ['swain', 'euclidean', 'chi2', 'correlation'],
            index=0,
            help="Méthode de calcul de similarité"
        )
        
        st.markdown("---")
        
        # Gestion du dataset
        st.subheader("📁 Gestion du Dataset")
        dataset_folder = st.text_input(
            "Chemin du dataset",
            value="dataset/Corel-1k",
            help="Chemin vers le dossier contenant les images"
        )
        
        if st.button("🧩 Construire l'index", use_container_width=True):
            if os.path.exists(dataset_folder):
                with st.spinner("Indexation en cours..."):
                    count = build_index(dataset_folder, bins, color_space)
                    st.success(f"✅ Index construit avec {count} images!")
            else:
                st.error("❌ Le dossier dataset n'existe pas!")
        
        st.markdown("---")
        
        # Upload d'image
        st.subheader("🖼️ Image Requête")
        uploaded_file = st.file_uploader(
            "Choisir une image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Sélectionnez une image pour la recherche"
        )
        
        # Information sur l'index
        if os.path.exists("descriptors.json"):
            try:
                with open("descriptors.json", "r") as f:
                    db = json.load(f)
                st.info(f"📊 Index disponible: {len(db)} images")
            except:
                pass

    # Contenu principal
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("🎯 Image Requête")
        
        # Afficher l'image sélectionnée
        query_image = None
        if uploaded_file is not None:
            query_image = Image.open(uploaded_file)
            st.image(query_image, caption="Image requête uploadée", use_container_width=True)
            
            # Métriques de l'image
            st.subheader("📊 Informations")
            col_metrics1, col_metrics2 = st.columns(2)
            
            with col_metrics1:
                st.metric("Dimensions", f"{query_image.size[0]}×{query_image.size[1]}")
            with col_metrics2:
                st.metric("Mode", query_image.mode)
            
        else:
            st.info("📸 Veuillez sélectionner une image requête dans la sidebar")
    
    with col2:
        st.header("📊 Analyse")
        
        if query_image is not None:
            # Histogrammes interactifs
            st.subheader("📈 Histogrammes")
            hist_fig = create_histogram_plot(query_image, color_space)
            st.plotly_chart(hist_fig, use_container_width=True)
            
            # Bouton de recherche
            if st.button("🔍 Lancer la recherche de similarité", use_container_width=True, type="primary"):
                if os.path.exists("descriptors.json"):
                    with st.spinner("Recherche des images similaires..."):
                        results = search(query_image, "descriptors.json", bins, color_space, method)
                        
                        if results:
                            st.session_state.results = results
                            st.session_state.query_image = query_image
                            st.session_state.dataset_folder = dataset_folder
                        else:
                            st.error("Aucun résultat trouvé!")
                else:
                    st.error("Veuillez d'abord construire l'index du dataset!")
    
    # Affichage des résultats
    if 'results' in st.session_state and st.session_state.results:
        st.markdown("---")
        st.header("🎯 Résultats de la Recherche")
        
        results = st.session_state.results
        query_image = st.session_state.query_image
        dataset_folder = st.session_state.dataset_folder
        
        # Métriques des résultats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Images trouvées", len(results))
        with col2:
            max_score = max(score for _, score in results)
            st.metric("Similarité max", f"{max_score:.3f}")
        with col3:
            min_score = min(score for _, score in results)
            st.metric("Similarité min", f"{min_score:.3f}")
        with col4:
            avg_score = np.mean([score for _, score in results])
            st.metric("Similarité moyenne", f"{avg_score:.3f}")
        
        # Graphique de distribution des scores
        scores = [score for _, score in results]
        fig_dist = px.histogram(
            x=scores, 
            nbins=10,
            title="Distribution des Scores de Similarité",
            labels={'x': 'Score de similarité', 'y': 'Nombre d\'images'}
        )
        fig_dist.update_layout(template='plotly_dark')
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Affichage des images résultats
        st.subheader(f"🖼️ Images Similaires ({len(results)} résultats)")
        
        # Organiser en grille responsive
        cols = st.columns(4)
        for idx, (img_path, score) in enumerate(results):
            with cols[idx % 4]:
                try:
                    full_path = os.path.join(dataset_folder, img_path)
                    if os.path.exists(full_path):
                        result_img = Image.open(full_path)
                        
                        # Déterminer la couleur du score
                        if score > 0.7:
                            score_color = "🟢"
                        elif score > 0.4:
                            score_color = "🟡"
                        else:
                            score_color = "🔴"
                        
                        st.image(
                            result_img, 
                            caption=f"{score_color} Score: {score:.3f}\n{os.path.basename(img_path)}",
                            use_container_width=True
                        )
                    else:
                        st.error(f"Image non trouvée: {img_path}")
                except Exception as e:
                    st.error(f"Erreur: {str(e)}")
        
        # Tableau détaillé des résultats
        st.subheader("📋 Détails des Résultats")
        results_df = pd.DataFrame(results, columns=['Image', 'Score'])
        results_df['Fichier'] = results_df['Image'].apply(os.path.basename)
        results_df = results_df[['Fichier', 'Score']].sort_values('Score', ascending=False)
        
        # Formater le tableau
        st.dataframe(
            results_df,
            use_container_width=True,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score de similarité",
                    help="Score de similarité avec l'image requête",
                    format="%.3f",
                    min_value=0,
                    max_value=1.0,
                )
            }
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "🔍 QueryPix CBIR System • Développé par Salma Barrak • 2025/2026"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()