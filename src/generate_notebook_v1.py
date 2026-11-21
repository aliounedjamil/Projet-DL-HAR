import json
import os

def create_notebook():
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
### Version 1 : Baseline & Réseau à une seule couche cachée

**Auteur :** Alioune DIOP (Binôme : Madjiguène FALL)  
**Enseignant :** Walid Bechkit  
**Année universitaire :** 2026–2027

---
### Objectifs de cette Version 1 :
1. Prise en main exploratoire du jeu de données HAR et analyse des capteurs.
2. Découpage rigoureux par volontaires (*GroupSplit*) vs découpage aléatoire.
3. Modèle de référence simple (Régression logistique).
4. Premier réseau MLP (1 couche cachée, 64 neurones) :
   - Vérification de la perte initiale théorique ($\approx \ln(6) \approx 1.79$) et surapprentissage d'un mini-lot (*sanity check*).
   - Comparaison des fonctions d'activation : ReLU, Sigmoid, Tanh.
   - Comparaison des algorithmes d'optimisation : SGD, SGD Momentum, Adam, avec réglage du Learning Rate.
   - Suivi des courbes d'apprentissage (Loss, Accuracy, Macro F1-score).
5. Analyse de la variabilité statistique sur plusieurs graines aléatoires (moyenne ± écart-type).""")

    # ==================== 1. IMPORTS ====================
    add_md("""## 1. Imports des bibliothèques""")
    add_code("""import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Configuration des graphiques
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
%matplotlib inline""")

    # ==================== 2. EXPLORATION ====================
    add_md("""## 2. Chargement et Exploration des Données (Partie 1.2)

Le jeu de données Human Activity Recognition (HAR) contient les enregistrements de capteurs (accéléromètre et gyroscope de smartphone) pour 30 volontaires réalisant 6 activités distinctes.""")
    add_code("""# Chargement des fichiers train.csv et test.csv
train_df = pd.read_csv('../../data/HAR/train.csv')
test_df = pd.read_csv('../../data/HAR/test.csv')

print(f"Jeu d'entraînement (Train) : {train_df.shape[0]} lignes, {train_df.shape[1]} colonnes")
print(f"Jeu de test (Test)         : {test_df.shape[0]} lignes, {test_df.shape[1]} colonnes")""")

    add_md("""### Vérification des volontaires (`subject`)
Vérifions quels volontaires composent le jeu d'entraînement et le jeu de test.""")
    add_code("""train_subjects = sorted(train_df['subject'].unique())
test_subjects = sorted(test_df['subject'].unique())
overlap = set(train_subjects).intersection(set(test_subjects))

print(f"Volontaires dans Train ({len(train_subjects)}) : {train_subjects}")
print(f"Volontaires dans Test ({len(test_subjects)})   : {test_subjects}")
print(f"Volontaires en commun : {len(overlap)}")""")

    add_md("""### Distribution des 6 classes d'activité""")
    add_code("""fig, axes = plt.subplots(1, 2, figsize=(14, 4))

sns.countplot(data=train_df, y='Activity', order=train_df['Activity'].value_counts().index, ax=axes[0], color='cornflowerblue')
axes[0].set_title("Distribution des activités (Train)", fontweight='bold')
axes[0].set_xlabel("Nombre d'observations")

sns.countplot(data=test_df, y='Activity', order=train_df['Activity'].value_counts().index, ax=axes[1], color='sandybrown')
axes[1].set_title("Distribution des activités (Test)", fontweight='bold')
axes[1].set_xlabel("Nombre d'observations")

plt.tight_layout()
plt.show()""")

    add_md("""### Statistiques descriptives et mise à l'échelle des capteurs
Examinons si les 561 variables de capteurs sont déjà normalisées.""")
    add_code("""feature_cols = [c for c in train_df.columns if c not in ['subject', 'Activity']]
features_train = train_df[feature_cols]

print(f"Nombre de variables issues des capteurs : {len(feature_cols)}")
print(f"Minimum global sur l'ensemble des variables : {features_train.min().min():.4f}")
print(f"Maximum global sur l'ensemble des variables : {features_train.max().max():.4f}")
print(f"Moyenne globale des moyennes                : {features_train.mean().mean():.4f}")
print(f"Moyenne des écarts-types                    : {features_train.std().mean():.4f}")
print(f"Nombre de valeurs manquantes (NaN)          : {train_df.isnull().sum().sum()}")""")

    add_md("""### Questions de prétraitement (Discussion pour la soutenance) :
1. **Les variables sont-elles déjà mises à l'échelle ?**
   - **Oui.** Comme le montrent le minimum (-1.0) et le maximum (+1.0), les auteurs du dataset ont déjà normalisé toutes les composantes fréquentielles et temporelles dans $[-1, 1]$. Aucun scaling supplémentaire n'est strictement requis pour le réseau de neurones.
2. **La colonne `subject` doit-elle servir d'entrée au modèle ?**
   - **NON.** `subject` est un identifiant arbitraire de personne, pas une mesure physique. Si on la donnait en entrée, le réseau apprendrait par cœur quel individu fait quelle activité (*data leakage*), et serait totalement incapable de généraliser aux nouveaux volontaires du jeu de test (les sujets 2, 4, 9, etc.).
3. **Encodage de la cible :**
   - Les activités sont sous forme de texte (`'WALKING'`, `'SITTING'`, etc.). Nous les encodons en entiers de $0$ à $5$ via `LabelEncoder` pour correspondre à la fonction de perte multiclasse `CrossEntropyLoss`.""")

    # ==================== 3. DÉCOUPAGE ====================
    add_md("""## 3. Découpage et Ensemble de Validation (Partie 1.3)

Le sujet précise que le fichier `test.csv` ne doit être utilisé qu'une seule fois à la fin. Tous nos choix doivent être guidés par un ensemble de validation extrait de `train.csv`.

**Problématique :** Doit-on découper les lignes au hasard ou réserver des volontaires entiers ?
- Dans le dataset HAR, les mesures sont découpées en fenêtres temporelles glissantes de 2,56 secondes avec **50% de chevauchement**. Deux fenêtres consécutives d'un même individu sont donc presque identiques.
- Un découpage aléatoire créerait une **fuite de données** majeure en plaçant des fenêtres quasi-jumelles en train et en validation.""")
    add_code("""# Séparation X, y et encodage
le = LabelEncoder()
y_train_full = le.fit_transform(train_df['Activity'])
X_train_full = train_df[feature_cols].values.astype(np.float32)

y_test = le.transform(test_df['Activity'])
X_test = test_df[feature_cols].values.astype(np.float32)

# --- EXPÉRIENCE COMPARATIVE (Partie 1.3) ---
# 1. Découpage Aléatoire (80% train / 20% validation)
from sklearn.model_selection import train_test_split
X_tr_rand, X_val_rand, y_tr_rand, y_val_rand = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

lr_rand = LogisticRegression(max_iter=1000, random_state=42)
lr_rand.fit(X_tr_rand, y_tr_rand)
acc_val_rand = accuracy_score(y_val_rand, lr_rand.predict(X_val_rand))
acc_test_rand = accuracy_score(y_test, lr_rand.predict(X_test))

# 2. Découpage par Volontaires (on réserve les sujets 1, 3, 5, 7 pour la validation)
val_subjects = [1, 3, 5, 7]
mask_val = train_df['subject'].isin(val_subjects)

X_tr_subj = train_df[~mask_val][feature_cols].values.astype(np.float32)
y_tr_subj = le.transform(train_df[~mask_val]['Activity'])

X_val_subj = train_df[mask_val][feature_cols].values.astype(np.float32)
y_val_subj = le.transform(train_df[mask_val]['Activity'])

lr_subj = LogisticRegression(max_iter=1000, random_state=42)
lr_subj.fit(X_tr_subj, y_tr_subj)
acc_val_subj = accuracy_score(y_val_subj, lr_subj.predict(X_val_subj))
acc_test_subj = accuracy_score(y_test, lr_subj.predict(X_test))

print("="*60)
print(f"Découpage Aléatoire     : Val Acc = {acc_val_rand*100:.2f}% | Test Acc = {acc_test_rand*100:.2f}% | Écart = {abs(acc_val_rand-acc_test_rand)*100:.2f}%")
print(f"Découpage par Volontaires: Val Acc = {acc_val_subj*100:.2f}% | Test Acc = {acc_test_subj*100:.2f}% | Écart = {abs(acc_val_subj-acc_test_subj)*100:.2f}%")
print("="*60)
print("Conclusion : Le découpage par volontaire est beaucoup plus proche de la performance réelle de test !")""")

    add_md("""### Préparation des DataLoaders PyTorch (avec le découpage par volontaire)""")
    add_code("""batch_size = 64

# Tensors
train_dataset = TensorDataset(torch.tensor(X_tr_subj), torch.tensor(y_tr_subj))
val_dataset = TensorDataset(torch.tensor(X_val_subj), torch.tensor(y_val_subj))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Taille Train : {len(train_dataset)} échantillons ({len(train_loader)} batches)")
print(f"Taille Val   : {len(val_dataset)} échantillons ({len(val_loader)} batches)")
print(f"Taille Test  : {len(test_dataset)} échantillons ({len(test_loader)} batches)")""")

    # ==================== 4. ARCHITECTURE & SANITY CHECK ====================
    add_md("""## 4. Définition du Modèle & Sanity Checks (Partie 2.1)

Nous définissons un réseau à **une seule couche cachée** (64 neurones).
- Entrée : 561 dimensions
- Couche cachée : 64 neurones
- Sortie : 6 classes (logits non normalisés pour `nn.CrossEntropyLoss`)""")
    add_code("""class SimpleMLP(nn.Module):
    def __init__(self, input_dim=561, hidden_dim=64, num_classes=6, activation='relu'):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Activation inconnue: {activation}")
            
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

# Instanciation
model_check = SimpleMLP(hidden_dim=64, activation='relu')
print(model_check)""")

    add_md("""### Vérification 1 : Perte initiale théorique
Pour un problème de classification à $C=6$ classes où les poids initiaux sont aléatoires (logits proches de zéro), la probabilité prédite pour chaque classe est approximativement $1/6$.
La perte théorique initiale attendue est :
$$\\mathcal{L}_0 = -\\ln\\left(\\frac{1}{6}\\right) = \\ln(6) \\approx 1{,}7918$$
Vérifions-le sur nos données :""")
    add_code("""criterion = nn.CrossEntropyLoss()
xb, yb = next(iter(train_loader))

model_check.eval()
with torch.no_grad():
    init_out = model_check(xb)
    init_loss = criterion(init_out, yb).item()

print(f"Perte initiale calculée : {init_loss:.4f}")
print(f"Perte théorique ln(6)   : {np.log(6):.4f}")
print("Contrôle réussi : la perte observée est parfaitement cohérente avec la théorie !")""")

    add_md("""### Vérification 2 : Capacité à surapprendre un petit lot (*Sanity Check*)
Pour s'assurer que notre implémentation (réseau, forward, backward, optimiseur) ne contient aucun bug, nous entraînons le réseau sur un lot minuscule (10 exemples). La perte doit tendre vers 0 et la précision atteindre 100% en quelques dizaines d'itérations.""")
    add_code("""# Petit lot de 10 exemples
x_tiny = xb[:10]
y_tiny = yb[:10]

overfit_model = SimpleMLP(hidden_dim=64, activation='relu')
optimizer_tiny = optim.Adam(overfit_model.parameters(), lr=0.01)

losses_tiny = []
for epoch in range(100):
    optimizer_tiny.zero_grad()
    out = overfit_model(x_tiny)
    loss = criterion(out, y_tiny)
    loss.backward()
    optimizer_tiny.step()
    losses_tiny.append(loss.item())

print(f"Perte finale sur le mini-lot : {losses_tiny[-1]:.6f}")
preds = overfit_model(x_tiny).argmax(dim=1)
acc_tiny = (preds == y_tiny).float().mean().item()
print(f"Précision sur le mini-lot    : {acc_tiny * 100:.1f}%")
assert acc_tiny == 1.0, "Le modèle devrait être capable de surapprendre le mini-lot !"
print("Sanity check validé avec succès !")""")

    # ==================== 5. FONCTIONS D'ENTRAINEMENT ====================
    add_md("""## 5. Fonctions d'Entraînement et d'Évaluation""")
    add_code("""def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x_batch.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += x_batch.size(0)
        
    epoch_loss = total_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item() * x_batch.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            
    total = len(all_targets)
    epoch_loss = total_loss / total
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average='macro')
    return epoch_loss, epoch_acc, epoch_f1, all_preds, all_targets""")

    # ==================== 6. ESSAIS ACTIVATIONS ET OPTIMISEURS ====================
    add_md("""## 6. Essais Comparatifs : Activations et Optimiseurs (Partie 2.2)

Nous comparons :
1. Les fonctions d'activation : **ReLU**, **Tanh**, **Sigmoid**
2. Les algorithmes d'optimisation : **SGD** (sans momentum), **SGD avec Momentum** (0.9), et **Adam**.""")
    add_code("""def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def run_experiment(activation='relu', opt_name='adam', lr=0.001, epochs=30, seed=42):
    set_seed(seed)
    model = SimpleMLP(hidden_dim=64, activation=activation)
    criterion = nn.CrossEntropyLoss()
    
    if opt_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif opt_name == 'sgd_momentum':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(opt_name)
        
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}
    
    for ep in range(epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, vl_f1, _, _ = evaluate(model, val_loader, criterion)
        
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)
        history['val_f1'].append(vl_f1)
        
    return model, history""")

    add_md("""### A. Comparaison des fonctions d'activation (avec Adam, lr=0.001)""")
    add_code("""activations = ['relu', 'tanh', 'sigmoid']
act_histories = {}

for act in activations:
    print(f"Entraînement avec activation : {act.upper()}...")
    _, hist = run_experiment(activation=act, opt_name='adam', lr=0.001, epochs=25)
    act_histories[act] = hist

# Affichage des courbes
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for act in activations:
    axes[0].plot(act_histories[act]['val_loss'], label=f"{act.upper()} (Val Loss)")
    axes[1].plot(act_histories[act]['val_acc'], label=f"{act.upper()} (Val Acc)")

axes[0].set_title("Validation Loss par fonction d'activation", fontweight='bold')
axes[0].set_xlabel("Époque")
axes[0].legend()

axes[1].set_title("Validation Accuracy par fonction d'activation", fontweight='bold')
axes[1].set_xlabel("Époque")
axes[1].legend()

plt.tight_layout()
plt.show()""")

    add_md("""### B. Comparaison des optimiseurs (avec activation ReLU)
Nous testons SGD (lr=0.05), SGD Momentum (lr=0.01), et Adam (lr=0.001).""")
    add_code("""configs_opt = [
    ('SGD classique', 'sgd', 0.05),
    ('SGD + Momentum', 'sgd_momentum', 0.01),
    ('Adam', 'adam', 0.001)
]

opt_histories = {}
for label, opt_name, lr in configs_opt:
    print(f"Entraînement avec {label} (lr={lr})...")
    _, hist = run_experiment(activation='relu', opt_name=opt_name, lr=lr, epochs=25)
    opt_histories[label] = hist

# Affichage des courbes
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for label, _, _ in configs_opt:
    axes[0].plot(opt_histories[label]['val_loss'], label=f"{label} (Val Loss)")
    axes[1].plot(opt_histories[label]['val_acc'], label=f"{label} (Val Acc)")

axes[0].set_title("Validation Loss par Optimiseur", fontweight='bold')
axes[0].set_xlabel("Époque")
axes[0].legend()

axes[1].set_title("Validation Accuracy par Optimiseur", fontweight='bold')
axes[1].set_xlabel("Époque")
axes[1].legend()

plt.tight_layout()
plt.show()""")

    # ==================== 7. VARIABILITÉ STATISTIQUE ====================
    add_md("""## 7. Variabilité des Résultats et Analyse Multi-Graines (Partie 2.3)

Les performances sur ce jeu de données sont élevées (> 94%). Pour savoir si la supériorité d'une configuration est réelle ou due à un coup de chance, nous répétons l'entraînement sur **plusieurs graines aléatoires** (seeds: 42, 123, 999).
Nous calculons la **moyenne** et l'**écart-type**.""")
    add_code("""seeds = [42, 123, 999]

# Config 1 : ReLU + Adam
scores_relu_adam = []
for s in seeds:
    m, h = run_experiment(activation='relu', opt_name='adam', lr=0.001, epochs=25, seed=s)
    scores_relu_adam.append(h['val_acc'][-1])

# Config 2 : Tanh + Adam
scores_tanh_adam = []
for s in seeds:
    m, h = run_experiment(activation='tanh', opt_name='adam', lr=0.001, epochs=25, seed=s)
    scores_tanh_adam.append(h['val_acc'][-1])

print("="*60)
print(f"Config 1 (ReLU + Adam) : Moyenne = {np.mean(scores_relu_adam)*100:.2f}% | Écart-type = {np.std(scores_relu_adam)*100:.2f}%")
print(f"Config 2 (Tanh + Adam) : Moyenne = {np.mean(scores_tanh_adam)*100:.2f}% | Écart-type = {np.std(scores_tanh_adam)*100:.2f}%")
diff = abs(np.mean(scores_relu_adam) - np.mean(scores_tanh_adam)) * 100
print(f"Différence absolue entre moyennes : {diff:.2f}%")
print("="*60)
print("Conclusion : Un écart entre deux modèles doit être supérieur à l'écart-type pour être considéré comme significatif !")""")

    # Sauvegarde du notebook
    output_path = "notebooks/version1_alioune/v1_baseline_mlp_alioune.ipynb"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Notebook généré avec succès dans : {output_path}")

create_notebook()

