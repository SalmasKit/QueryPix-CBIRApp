# cbir_logic.py

import os
import cv2
import json
import numpy as np
import streamlit as st
from PIL import Image
from plotly.subplots import make_subplots
import plotly.graph_objects as go

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
                
                progress_bar.progress((count / total_images))
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

    # Convert PIL → CV2
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
        
        if i % 50 == 0:
            progress_bar.progress(i / total)
            status_text.text(f"Recherche: {i}/{total} images...")

    results.sort(key=lambda x: x[1], reverse=True)
    progress_bar.empty()
    status_text.empty()

    return results[:12]

def create_histogram_plot(query_image, color_space='RGB'):
    """Crée un graphique d'histogramme interactif avec Plotly."""
    img_array = np.array(query_image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_array = cv2.resize(img_array, (256, 256))
    
    hist_full = hist_features(img_array, bins=256, color_space=color_space)
    hist_bin = hist_features(img_array, bins=16, color_space=color_space)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Histogramme Complet (256 bins)', 'Histobine (16 bins)'),
        horizontal_spacing=0.1
    )
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    names = ['Canal 1', 'Canal 2', 'Canal 3']
    
    for i in range(3):
        fig.add_trace(
            go.Scatter(
                x=list(range(256)),
                y=hist_full[i*256:(i+1)*256],
                mode='lines',
                line=dict(color=colors[i], width=2),
            ),
            row=1, col=1
        )
    
    for i in range(3):
        fig.add_trace(
            go.Scatter(
                x=list(range(16)),
                y=hist_bin[i*16:(i+1)*16],
                mode='lines+markers',
                line=dict(color=colors[i], width=2),
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        height=400,
        template='plotly_dark',
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig
