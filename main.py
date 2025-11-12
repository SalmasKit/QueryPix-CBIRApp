"""
CBIR simple - Corel-1000
Auteur : Salma Barrak
Année : 2025/2026
Ce code calcule les histogrammes des images, les indexe, et permet
de rechercher les plus similaires via une interface Tkinter moderne.
"""

import os
import cv2
import json
import numpy as np
import math
from matplotlib import pyplot as plt
from tkinter import filedialog, Tk, StringVar, IntVar, messagebox
import tkinter as tk

# ==========================
# FONCTIONS HISTOGRAMMES
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


# ==========================
# AFFICHAGE DES HISTOGRAMMES
# ==========================
def show_histograms(image_path, color_space='RGB'):
    """Affiche l’histogramme complet et l’histobine."""
    image = cv2.imread(image_path)
    if image is None:
        messagebox.showerror("Erreur", "Impossible de charger l'image.")
        return

    image = cv2.resize(image, (256, 256))

    plt.figure(figsize=(12, 5))

    # Histogramme complet (256 bins)
    hist_full = hist_features(image, bins=256, color_space=color_space)
    plt.subplot(1, 2, 1)
    for i, color in enumerate(['r', 'g', 'b']):
        plt.plot(hist_full[i*256:(i+1)*256], color=color)
    plt.title("Histogramme complet (256 bins)")

    # Histobine (16 bins)
    hist_bin = hist_features(image, bins=16, color_space=color_space)
    plt.subplot(1, 2, 2)
    for i, color in enumerate(['r', 'g', 'b']):
        plt.plot(hist_bin[i*16:(i+1)*16], color=color)
    plt.title("Histobine (16 bins)")

    plt.tight_layout()
    plt.show()


# ==========================
# INDEXATION DU DATASET
# ==========================
def build_index(dataset_folder="dataset/Corel-1k", output_file="descriptors.json",
                bins=16, color_space='RGB'):
    """Construit et sauvegarde les descripteurs du dataset."""
    descriptors = {}
    count = 0
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
                if count % 100 == 0:
                    print(f"Indexé : {count} images...")

    with open(output_file, "w") as f:
        json.dump(descriptors, f, indent=2)
    print(f"✅ Index sauvegardé ({count} images).")


# ==========================
# MESURES DE DISTANCE
# ==========================
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


# ==========================
# RECHERCHE D’IMAGES SIMILAIRES
# ==========================
def search(query_path, descriptors_file="descriptors.json",
           bins=16, color_space='RGB', method='swain'):
    """Recherche les images les plus similaires."""
    if not os.path.exists(descriptors_file):
        messagebox.showerror("Erreur", "Le fichier descriptors.json n'existe pas.")
        return []

    with open(descriptors_file, "r") as f:
        db = json.load(f)

    query_img = cv2.imread(query_path)
    if query_img is None:
        messagebox.showerror("Erreur", "Impossible de lire l'image requête.")
        return []

    query_img = cv2.resize(query_img, (256, 256))
    query_features = hist_features(query_img, bins, color_space)

    results = []
    for rel_path, desc in db.items():
        desc_array = np.array(desc)
        score = distance(query_features, desc_array, method)
        results.append((rel_path, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:10]


# ==========================
# AFFICHAGE DES RÉSULTATS
# ==========================
def show_results(query_path, results, dataset_folder="dataset/Corel-1k"):
    """Affiche les résultats visuellement."""
    if not results:
        messagebox.showwarning("Avertissement", "Aucun résultat trouvé.")
        return

    images_per_row = 5
    total = len(results) + 1
    rows = math.ceil(total / images_per_row)
    plt.figure(figsize=(3 * images_per_row, 3 * rows))

    # Image requête
    plt.subplot(rows, images_per_row, 1)
    plt.imshow(cv2.cvtColor(cv2.imread(query_path), cv2.COLOR_BGR2RGB))
    plt.title("Image requête")
    plt.axis("off")

    # Images similaires
    for i, (rel_path, score) in enumerate(results):
        plt.subplot(rows, images_per_row, i + 2)
        img = cv2.imread(os.path.join(dataset_folder, rel_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(f"{rel_path}\nScore: {score:.3f}", fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# ==========================
# INTERFACE TKINTER MODERNE
# ==========================
def open_gui(dataset_folder="dataset/Corel-1k", bins=16, color_space='RGB'):
    root = Tk()
    root.title("CBIR - Corel-1000")
    root.geometry("550x400")
    root.configure(bg="#f5f6f7")

    # Variables
    method_var = StringVar(value='swain')
    color_var = StringVar(value=color_space)
    bins_var = IntVar(value=bins)

    # Titre
    tk.Label(root, text="Système CBIR - Corel-1000",
             font=("Segoe UI", 16, "bold"), bg="#f5f6f7", fg="#333").pack(pady=10)

    frame = tk.Frame(root, bg="white", bd=2, relief="ridge")
    frame.pack(padx=20, pady=10, fill="both", expand=True)

    # Options utilisateur
    tk.Label(frame, text="Espace couleur :", bg="white", font=("Segoe UI", 11)).grid(row=0, column=0, padx=10, pady=10, sticky="w")
    tk.OptionMenu(frame, color_var, 'RGB', 'HSV', 'Lab').grid(row=0, column=1, padx=10)

    tk.Label(frame, text="Nombre de bins :", bg="white", font=("Segoe UI", 11)).grid(row=1, column=0, padx=10, pady=10, sticky="w")
    tk.OptionMenu(frame, bins_var, 8, 16, 32, 64, 256).grid(row=1, column=1, padx=10)

    tk.Label(frame, text="Distance :", bg="white", font=("Segoe UI", 11)).grid(row=2, column=0, padx=10, pady=10, sticky="w")
    tk.OptionMenu(frame, method_var, 'swain', 'euclidean', 'chi2', 'correlation').grid(row=2, column=1, padx=10)

    # Boutons
    def build():
        build_index(dataset_folder, "descriptors.json", bins_var.get(), color_var.get())
        messagebox.showinfo("Indexation", "Indexation terminée avec succès.")

    def browse():
        path = filedialog.askopenfilename(title="Choisir une image requête")
        if path:
            show_histograms(path, color_var.get())
            results = search(path, "descriptors.json", bins_var.get(),
                             color_var.get(), method_var.get())
            show_results(path, results, dataset_folder)

    tk.Button(frame, text="🧩 Construire l'index", font=("Segoe UI", 11, "bold"),
              bg="#4CAF50", fg="white", command=build).grid(row=3, column=0, pady=20, ipadx=10)
    tk.Button(frame, text="🔍 Rechercher image", font=("Segoe UI", 11, "bold"),
              bg="#2196F3", fg="white", command=browse).grid(row=3, column=1, pady=20, ipadx=10)

    tk.Label(root, text="💡 Astuce : Essayez différentes distances ou espaces couleur.",
             bg="#f5f6f7", fg="#555", font=("Segoe UI", 9, "italic")).pack(pady=5)

    root.mainloop()


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    dataset_folder = "dataset/Corel-1k"
    descriptors_file = "descriptors.json"
    bins = 16
    color_space = 'RGB'

    open_gui(dataset_folder, bins, color_space)
