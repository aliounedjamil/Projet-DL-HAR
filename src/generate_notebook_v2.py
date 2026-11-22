import json
import os

def create_v2_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_md(text):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.split("\n")]
        })

    def add_code(code):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code.split("\n")]
        })

    # ==================== TITRE & AUTEURS ====================
    add_md("""# M2 MALIA - Deep Learning
## Projet HAR : Étude progressive des réseaux de neurones
### Version 2 : Réseau Profond, Régularisation, Interprétabilité & Évaluation Finale

**Auteur :** Madjiguène FALL (Binôme : Alioune DIOP)  
**Enseignant :** Walid Bechkit  
**Année universitaire :** 2026–2027

---
### Objectifs de cette Version 2 :
1. **Extension de l'architecture (Partie 3)** :
   - Élargissement de la couche cachée (128, 256 neurones) et étude de l'overfitting.
   - Implémentation de réseaux profonds (2 à 4 couches cachées), calcul du nombre de paramètres et temps de calcul.
2. **Régularisation et robustesse (Partie 4)** :
   - Dropout ($p \in \{0.1, 0.3, 0.5\}$), Weight Decay (L2), Batch Normalization, et Early Stopping.
   - Sélection des 2 ou 3 configurations les plus performantes sur validation.
3. **Visualisation et Évaluation Finale (Partie 5)** :
   - Tracé des courbes d'apprentissage (loss et macro-F1).
   - Évaluation unique sur le jeu de test final (`test.csv`).
   - Matrice de confusion et analyse physique des erreurs de capteurs (*sitting* vs *standing*).
4. **Interprétabilité et Réduction de Dimension (Partie 6)** :
   - Projection t-SNE / UMAP comparant les variables brutes à l'espace latent de la dernière couche cachée.""")

    # ==================== 1. IMPORTS ====================
    add_md("""## 1. Imports des bibliothèques et modules du projet""")
    add_code("""import os
import sys
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import des fonctions partagées créées par Alioune dans src/
sys.path.append('../../src')
from data_loader import load_har_data, split_by_subject, prepare_features_and_labels, create_dataloaders

# Configuration graphique
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
%matplotlib inline""")

    # ==================== 2. CHARGEMENT ====================
    add_md("""## 2. Chargement des Données et DataLoaders
Nous réutilisons le découpage validé par volontaires entiers (sujets 1, 3, 5, 7 réservés pour la validation).""")
    add_code("""train_df, test_df = load_har_data('../../data/HAR')

# Découpage par volontaires
val_subjects = [1, 3, 5, 7]
train_split, val_split = split_by_subject(train_df, val_subjects=val_subjects)

# Préparation X, y
X_tr, y_tr, le, feature_cols = prepare_features_and_labels(train_split)
X_val, y_val, _, _ = prepare_features_and_labels(val_split, label_encoder=le)
X_test, y_test, _, _ = prepare_features_and_labels(test_df, label_encoder=le)

batch_size = 64
train_loader, val_loader = create_dataloaders(X_tr, y_tr, X_val, y_val, batch_size=batch_size)
_, test_loader = create_dataloaders(X_tr, y_tr, X_test, y_test, batch_size=batch_size)

print(f"Échantillons d'entraînement : {len(X_tr)}")
print(f"Échantillons de validation   : {len(X_val)}")
print(f"Échantillons de test final   : {len(X_test)}")
print(f"Classes d'activité           : {list(le.classes_)}")""")

    # ==================== 3. ARCHITECTURE PROFONDE ====================
    add_md("""## 3. Définition de l'Architecture Profonde & Régularisée (Parties 3 & 4)

Cette classe permet de créer un MLP profond modulaire avec :
- Une liste de dimensions cachées (ex: `[256, 128, 64]`)
- L'activation choisie (ReLU, Tanh, etc.)
- La Batch Normalization optionnelle (`use_bn=True`)
- Le Dropout configurable (`dropout_p=0.3`)""")
    add_code("""class DeepMLP(nn.Module):
    def __init__(self, input_dim=561, hidden_dims=[128, 64], num_classes=6, 
                 activation='relu', dropout_p=0.0, use_bn=False):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(h_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            prev_dim = h_dim
            
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)
        
    def forward(self, x):
        feat = self.features(x)
        out = self.classifier(feat)
        return out
    
    def extract_features(self, x):
        \"\"\"Extrait les représentations latentes de la dernière couche cachée pour t-SNE / UMAP\"\"\"
        return self.features(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Test d'instanciation
test_net = DeepMLP(hidden_dims=[256, 128, 64], dropout_p=0.3, use_bn=True)
print(test_net)
print(f"Nombre total de paramètres entraînables : {count_parameters(test_net):,}")""")

    # ==================== 4. BOUCLE ENTRAINEMENT & EARLY STOPPING ====================
    add_md("""## 4. Fonctions d'Entraînement avec Early Stopping""")
    add_code("""def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x_b, y_b in dataloader:
        optimizer.zero_grad()
        out = model(x_b)
        loss = criterion(out, y_b)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x_b.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y_b).sum().item()
        total += x_b.size(0)
    return total_loss / total, correct / total

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_b, y_b in dataloader:
            out = model(x_b)
            loss = criterion(out, y_b)
            total_loss += loss.item() * x_b.size(0)
            all_preds.extend(out.argmax(dim=1).cpu().numpy())
            all_targets.extend(y_b.cpu().numpy())
            
    total = len(all_targets)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    return total_loss / total, acc, macro_f1, all_preds, all_targets

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=40, patience=7):
    \"\"\"Entraîne le modèle avec suivi des courbes et Early Stopping\"\"\"
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    start_time = time.time()
    for ep in range(epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, vl_f1, _, _ = evaluate(model, val_loader, criterion)
        
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)
        history['val_f1'].append(vl_f1)
        
        # Early Stopping
        if vl_loss < best_loss:
            best_loss = vl_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping déclenché à l'époque {ep+1} !")
                break
                
    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.time() - start_time
    return model, history, elapsed""")

    # ==================== 5. EXTENSION ARCHITECTURE ====================
    add_md("""## 5. Extension de l'Architecture : Largeur & Profondeur (Partie 3)
*À exécuter et analyser par Madjiguène : comparaison des largeurs (128, 256) et des profondeurs (2 à 4 couches).*""")
    add_code("""# Configuration des architectures à tester
architectures = {
    '1 couche (64) [Baseline V1]': [64],
    '1 couche (128) [Plus large]': [128],
    '1 couche (256) [Très large]': [256],
    '2 couches [128, 64]': [128, 64],
    '3 couches [256, 128, 64]': [256, 128, 64],
    '4 couches [256, 128, 64, 32]': [256, 128, 64, 32]
}

results_arch = []
for name, dims in architectures.items():
    set_seed(42)
    m = DeepMLP(hidden_dims=dims, activation='relu')
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(m.parameters(), lr=0.001)
    
    m, hist, duration = train_model(m, train_loader, val_loader, crit, opt, epochs=30, patience=6)
    n_params = count_parameters(m)
    best_val_acc = max(hist['val_acc']) * 100
    best_val_f1 = max(hist['val_f1']) * 100
    
    results_arch.append({
        'Architecture': name,
        'Paramètres': n_params,
        'Temps (s)': round(duration, 2),
        'Best Val Acc (%)': round(best_val_acc, 2),
        'Best Val F1 (%)': round(best_val_f1, 2)
    })

df_arch = pd.DataFrame(results_arch)
print(df_arch.to_string(index=False))""")

    # ==================== 6. REGULARISATION ====================
    add_md("""## 6. Régularisation et Robustesse (Partie 4)
*À compléter par Madjiguène : impact du Dropout (0.1, 0.3, 0.5), Weight Decay (L2) et Batch Normalization.*""")
    add_code("""# Exemple : Test de Dropout et Batch Normalization sur l'architecture 3 couches
configs_reg = [
    ('Sans régularisation', 0.0, False, 0.0),
    ('Dropout 0.3', 0.3, False, 0.0),
    ('Dropout 0.5', 0.5, False, 0.0),
    ('Weight Decay (1e-4)', 0.0, False, 1e-4),
    ('BatchNorm + Dropout 0.3', 0.3, True, 0.0),
    ('BatchNorm + Dropout 0.3 + Weight Decay', 0.3, True, 1e-4)
]

# À tester et analyser pour sélectionner le meilleur modèle !""")

    # ==================== 7. EVALUATION TEST & CONFUSION ====================
    add_md("""## 7. Évaluation Finale sur Test & Matrice de Confusion (Partie 5)
⚠️ **RÈGLE CRUCIALE DU SUJET :** Le fichier de test ne doit servir qu'une seule fois, pour l'évaluation finale des 2 ou 3 configurations retenues.""")
    add_code("""# À exécuter à la fin sur le test_loader pour le meilleur modèle retenu !
# _, test_acc, test_f1, preds_test, targets_test = evaluate(best_model, test_loader, criterion)
# cm = confusion_matrix(targets_test, preds_test)
# sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')""")

    # ==================== 8. INTERPRETABILITE ====================
    add_md("""## 8. Interprétabilité et Réduction de Dimension : t-SNE / UMAP (Partie 6)
Visualisation comparative :
1. Sur les variables d'entrée brutes (561 capteurs)
2. Sur les représentations latentes de la dernière couche cachée.""")
    add_code("""# À implémenter avec TSNE(n_components=2, perplexity=30, random_state=42)
# pour visualiser l'organisation des classes dans l'espace latent""")

    # Sauvegarde
    out_path = "notebooks/version2_madjiguene/v2_deep_mlp_madjiguene.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Notebook V2 prêt pour Madjiguène généré dans : {out_path}")

create_v2_notebook()

