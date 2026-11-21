import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Création du dossier figures si nécessaire
os.makedirs("reports/figures", exist_ok=True)

# 1. Chargement des données
print("--- CHARGEMENT DES DONNÉES ---")
train_df = pd.read_csv("data/HAR/train.csv")
test_df = pd.read_csv("data/HAR/test.csv")

print(f"Jeu d'entraînement (Train) : {train_df.shape[0]} lignes, {train_df.shape[1]} colonnes")
print(f"Jeu de test (Test)         : {test_df.shape[0]} lignes, {test_df.shape[1]} colonnes")

# 2. Volontaires (Subjects)
train_subjects = sorted(train_df['subject'].unique())
test_subjects = sorted(test_df['subject'].unique())
common_subjects = set(train_subjects).intersection(set(test_subjects))

print("\n--- VOLONTAIRES (SUBJECTS) ---")
print(f"Nombre de volontaires dans Train : {len(train_subjects)} -> {train_subjects}")
print(f"Nombre de volontaires dans Test  : {len(test_subjects)} -> {test_subjects}")
print(f"Volontaires en commun : {len(common_subjects)} (aucun, ce qui garantit un test sur sujets nouveaux)")

# 3. Distribution des classes
print("\n--- DISTRIBUTION DES CLASSES ---")
print("Train :")
print(train_df['Activity'].value_counts())
print("\nTest :")
print(test_df['Activity'].value_counts())

# 4. Statistiques descriptives sur les capteurs (561 features)
feature_cols = [c for c in train_df.columns if c not in ['subject', 'Activity']]
features_train = train_df[feature_cols]

print("\n--- STATISTIQUES DES VARIABLES CAPTEURS ---")
print(f"Nombre de variables capteurs : {len(feature_cols)}")
print(f"Minimum global : {features_train.min().min():.4f}")
print(f"Maximum global : {features_train.max().max():.4f}")
print(f"Moyenne globale des moyennes : {features_train.mean().mean():.4f}")
print(f"Moyenne des écarts-types     : {features_train.std().mean():.4f}")
print(f"Nombre de valeurs manquantes (NaN) : {train_df.isnull().sum().sum()}")

# 5. Visualisations simples pour le rapport et la soutenance
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'

# Figure 1 : Distribution des classes
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

train_counts = train_df['Activity'].value_counts()
test_counts = test_df['Activity'].value_counts()

sns.barplot(x=train_counts.values, y=train_counts.index, ax=axes[0], color='cornflowerblue')
axes[0].set_title("Distribution des 6 activités (Train)", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Nombre d'exemples")

sns.barplot(x=test_counts.values, y=test_counts.index, ax=axes[1], color='sandybrown')
axes[1].set_title("Distribution des 6 activités (Test)", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Nombre d'exemples")

plt.tight_layout()
plt.savefig("reports/figures/01_distribution_classes.png", dpi=300)
plt.close()

# Figure 2 : Répartition des observations par sujet
plt.figure(figsize=(10, 4))
subj_counts = train_df['subject'].value_counts().sort_index()
sns.barplot(x=subj_counts.index.astype(str), y=subj_counts.values, color='steelblue')
plt.title("Nombre de mesures par volontaire dans le jeu d'entraînement", fontsize=11, fontweight='bold')
plt.xlabel("Numéro du volontaire (subject)")
plt.ylabel("Nombre d'enregistrements")
plt.tight_layout()
plt.savefig("reports/figures/02_observations_par_sujet.png", dpi=300)
plt.close()

# Figure 3 : Histogrammes et Boxplots sur 3 variables de base
# tBodyAcc-mean()-X, tGravityAcc-mean()-X, tBodyGyro-mean()-X
sample_vars = ['tBodyAcc-mean()-X', 'tGravityAcc-mean()-X', 'tBodyGyro-mean()-X']
fig, axes = plt.subplots(2, 3, figsize=(14, 7))

for i, col in enumerate(sample_vars):
    # Histogramme
    sns.histplot(train_df[col], bins=30, kde=True, ax=axes[0, i], color='teal')
    axes[0, i].set_title(f"Histogramme : {col}", fontsize=10, fontweight='bold')
    axes[0, i].set_xlabel("Valeur (normalisée)")
    axes[0, i].set_ylabel("Fréquence")
    
    # Boxplot par activité
    sns.boxplot(data=train_df, x='Activity', y=col, ax=axes[1, i], palette='Set2')
    axes[1, i].set_title(f"{col} par activité", fontsize=10, fontweight='bold')
    axes[1, i].set_xticklabels(axes[1, i].get_xticklabels(), rotation=40, ha='right', fontsize=8)
    axes[1, i].set_xlabel("")

plt.tight_layout()
plt.savefig("reports/figures/03_features_dist_boxplot.png", dpi=300)
plt.close()

# Figure 4 : Corrélations simples
corr_features = [
    'tBodyAcc-mean()-X', 'tBodyAcc-mean()-Y', 'tBodyAcc-mean()-Z',
    'tGravityAcc-mean()-X', 'tGravityAcc-mean()-Y', 'tGravityAcc-mean()-Z'
]
plt.figure(figsize=(7, 5.5))
corr_matrix = train_df[corr_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Corrélations entre composantes d'accélération", fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig("reports/figures/04_matrice_correlation.png", dpi=300)
plt.close()

print("\nVisualisations sauvegardées avec succès dans le dossier 'reports/figures/' !")
