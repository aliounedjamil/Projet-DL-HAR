# M2 MALIA - Deep Learning : Projet HAR (Human Activity Recognition)

**Auteurs (Binôme) :**
- **Alioune DIOP**
- **Madjiguène FALL**

**Enseignant :** Walid Bechkit  
**Année universitaire :** 2026–2027  
**Dépôt GitHub :** [https://github.com/aliounedjamil/Projet-DL-HAR](https://github.com/aliounedjamil/Projet-DL-HAR)

---

## 🏆 Statut du Projet : 100% Terminé & Validé ✅

L'ensemble des exigences du sujet a été entièrement réalisé, testé et consigné dans les deux notebooks et le rapport scientifique de 4 pages :

```text
====================================================================================
                        BILAN COMPLET DU PROJET HAR (M2 MALIA)
====================================================================================

👤 ALIOUNE DIOP (Parties 1 & 2 - Version 1)       👤 MADJIGUÈNE FALL (Parties 3 à 7 - Version 2)
[ Statut : ✅ TERMINÉ & EXÉCUTÉ ]                 [ Statut : ✅ TERMINÉ & EXÉCUTÉ ]
------------------------------------------------------------------------------------
1. Exploration & Prétraitements (Partie 1.2)     1. Extension d'architecture (Partie 3)
   - 7 352 train / 2 947 test, 561 capteurs         - Largeur (64, 128, 256) & Profondeur (1 à 4 couches)
   - Normalisation [-1, 1], exclusion subject       - Analyse nombre de paramètres & temps de calcul
2. Méthodologie Découpage (Partie 1.3)           2. Régularisation & Robustesse (Partie 4)
   - Split par volontaires vs Aléatoire             - Dropout (0.1, 0.3, 0.5), Weight Decay (L2)
   - Preuve de fidélité au test (0.71% vs 2.44%)    - Batch Normalization, Early Stopping (96.76% val)
3. Modèle de référence (Baseline - Partie 1.4)   3. Évaluation Test & Confusion (Partie 5)
   - Régression Logistique (95.99% val / 95.28% test) - Passage UNIQUE sur test.csv (93.89% test)
4. MLP 1 couche cachée (Partie 2.1)                 - Matrice de confusion & analyse physique
   - Perte théorique ln(6) ~ 1.79 vérifiée       4. Interprétabilité t-SNE (Partie 6)
   - Sanity check mini-lot réussi à 100%            - Projection 2D : Espace brut vs Espace latent
5. Activations & Optimiseurs (Partie 2.2)        5. Bonus / Ouverture (Partie 7)
   - ReLU vs Tanh vs Sigmoid                        - Étude d'ablation des familles de capteurs
   - SGD vs SGD+Momentum vs Adam                    (rôle fondamental de la composante gravitationnelle)
6. Variabilité statistique (Partie 2.3)          6. Livrables Finaux (Partie 8)
   - Répétition sur seeds 42, 123, 999              - Rapport PDF de 4 pages (M2_DL_2026_DIOP_FALL.pdf)
   - Moyenne ± écart-type (Tanh + Adam : 95.53%)    - Archive Moodle finale (M2_DL_2026_DIOP_FALL.zip)
====================================================================================
```

---

## 📁 Arborescence Complète du Projet

```text
Projet DL/
├── .gitignore
├── requirements.txt
├── README.md
├── M2_DL_2026_DIOP_FALL.zip               # 📦 Archive finale complète pour Moodle
├── data/
│   └── HAR/
│       ├── train.csv                      # Données d'entraînement (7 352 lignes, 563 colonnes)
│       ├── test.csv                       # Données de test (2 947 lignes, 563 colonnes)
│       └── .gitkeep
├── src/                                   # Code Python modulaire et reproductible
│   ├── __init__.py
│   ├── data_loader.py                     # Chargement et découpage par volontaires
│   ├── train_v1_baseline.py               # Entraînement complet Version 1 (MLP 1 couche)
│   ├── train_v2_deep_mlp.py               # Entraînement complet Version 2 (MLP profond, régularisation, t-SNE)
│   ├── experiment_split_comparison.py     # Expérience de découpage par sujet vs aléatoire
│   ├── generate_exploratory_figures.py    # Génération des figures 01 à 04
│   ├── generate_pdf_report.py             # Compilation du rapport scientifique en PDF (ReportLab)
│   ├── populate_notebook_outputs.py       # Injection des outputs du Notebook V1
│   ├── populate_notebook_v2_outputs.py    # Injection des outputs du Notebook V2
│   └── create_final_submission_zip.py     # Génération de l'archive ZIP Moodle
├── notebooks/
│   ├── version1_alioune/                  # 👤 Version 1 : Alioune DIOP
│   │   ├── README.md
│   │   └── v1_baseline_mlp_alioune.ipynb  # Notebook 100% exécuté avec figures intégrées
│   └── version2_madjiguene/               # 👤 Version 2 : Madjiguène FALL
│       ├── README.md
│       └── v2_deep_mlp_madjiguene.ipynb   # Notebook 100% exécuté avec figures intégrées
├── reports/
│   ├── M2_DL_2026_DIOP_FALL.pdf           # 📄 Rapport scientifique final (4 pages, conforme max 6)
│   ├── rapport_partie_alioune.md          # Brouillon des sections 1 à 3
│   ├── README.md
│   └── figures/                           # 12 figures haute résolution
│       ├── 01_distribution_classes.png
│       ├── 02_observations_par_sujet.png
│       ├── 03_features_dist_boxplot.png
│       ├── 04_matrice_correlation.png
│       ├── 05_comparaison_activations.png
│       ├── 06_comparaison_optimiseurs.png
│       ├── 07_comparaison_largeur_profondeur.png
│       ├── 08_comparaison_regularisation.png
│       ├── 09_courbes_apprentissage_meilleur_modele.png
│       ├── 10_matrice_confusion_test.png
│       ├── 11_tsne_brut_vs_latent.png
│       └── 12_importance_capteurs_ablation.png
└── deliverables/
    ├── M2_DL_2026_DIOP_FALL.zip           # Archive zippée Moodle
    └── Projet_DL_Alioune_DIOP.zip
```

---

## 🚀 Reproduction des Résultats

1. **Cloner le dépôt :**
   ```bash
   git clone https://github.com/aliounedjamil/Projet-DL-HAR.git
   cd Projet-DL-HAR
   ```

2. **Installer les dépendances :**
   ```bash
   pip install -r requirements.txt
   ```

3. **Exécuter les scripts de reproduction :**
   ```bash
   # 1. Exploration et statistiques descriptives (Partie 1.2)
   python src/generate_exploratory_figures.py

   # 2. Expérience comparative de découpage (Partie 1.3 & 1.4)
   python src/experiment_split_comparison.py

   # 3. Entraînement de la Version 1 (Partie 2)
   python src/train_v1_baseline.py

   # 4. Entraînement complet de la Version 2 (Parties 3 à 7)
   python src/train_v2_deep_mlp.py

   # 5. Recompilation du rapport scientifique PDF (Partie 8.2)
   python src/generate_pdf_report.py
   ```
