# M2 MALIA - Deep Learning : Projet HAR (Human Activity Recognition)

**Auteurs (Binôme) :**
- Alioune DIOP
- Madjiguène FALL

**Enseignant :** Walid Bechkit  
**Année universitaire :** 2026–2027

---

## 📌 Présentation du Projet
Ce projet vise à construire et à analyser différentes architectures de réseaux de neurones multicouches (MLP) pour un problème de classification d'activités humaines à partir de signaux de capteurs (accéléromètres et gyroscopes de smartphones - HAR Dataset).

Le projet s'attache à comprendre l'influence de :
- La complexité de l'architecture (largeur, profondeur) ;
- Le choix des fonctions d'activation (ReLU, Tanh, Sigmoid) ;
- Les algorithmes d'optimisation (SGD, SGD Momentum, Adam) et le pas d'apprentissage ;
- Les techniques de régularisation (Dropout, BatchNorm, Weight Decay, Early Stopping) ;
- La variabilité statistique (évaluation répétée sur plusieurs graines aléatoires) ;
- L'interprétabilité des représentations latentes (t-SNE / UMAP) et l'analyse physique des erreurs.

---

## 📁 Structure du Projet

```text
Projet DL/
├── .gitignore
├── requirements.txt
├── README.md
├── data/
│   └── HAR/
│       ├── train.csv                      # Données d'entraînement (7352 lignes, 563 colonnes)
│       ├── test.csv                       # Données de test (2947 lignes, 563 colonnes)
│       └── .gitkeep
├── src/                                   # Code Python partagé et réutilisable
│   ├── __init__.py
│   ├── data_loader.py                     # Chargement et découpage par volontaire
│   ├── models.py                          # Architectures MLP et baseline
│   ├── train.py                           # Fonctions train_epoch et evaluate
│   ├── evaluate.py                        # Métriques (F1, Accuracy, Matrice de confusion)
│   └── utils.py                           # Reproductibilité (graines) et tracés
├── notebooks/
│   ├── version1_alioune/                  # Version 1 : Baseline & Réseau 1 couche
│   │   ├── README.md
│   │   └── v1_baseline_mlp_alioune.ipynb
│   └── version2_madjiguene/               # Version 2 : Réseau profond & Interprétabilité
│       ├── README.md
│       └── v2_deep_mlp_madjiguene.ipynb
├── reports/                               # Rapport scientifique (max 6 pages)
│   ├── figures/                           # Graphiques et visualisations générées
│   ├── README.md
│   └── M2_DL_2026_DIOP_FALL.pdf           # Rapport PDF final
└── deliverables/                          # Archive finale pour soumission Moodle
    └── M2_DL_2026_DIOP_FALL.zip
```

---

## 🚀 Installation & Utilisation

1. **Cloner le dépôt :**
   ```bash
   git clone <URL_DU_DEPOT>
   cd "Projet DL"
   ```

2. **Créer un environnement virtuel et installer les dépendances :**
   ```bash
   python -m venv venv
   # Sur Windows :
   venv\Scripts\activate
   # Sur Linux / Mac :
   source venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Lancer Jupyter Notebook :**
   ```bash
   jupyter notebook
   ```
