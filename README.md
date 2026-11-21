# M2 MALIA - Deep Learning : Projet HAR (Human Activity Recognition)

**Auteurs (Binôme) :**
- **Alioune DIOP**
- **Madjiguène FALL**

**Enseignant :** Walid Bechkit  
**Année universitaire :** 2026–2027  
**Dépôt GitHub :** [https://github.com/aliounedjamil/Projet-DL-HAR](https://github.com/aliounedjamil/Projet-DL-HAR)

---

## 👥 Répartition Détaillée des Tâches (Binôme 50% / 50%)

Pour garantir une collaboration équitable et efficace, le travail a été scindé en deux moitiés complémentaires conformément aux consignes du sujet (deux versions de notebooks et un rapport de 6 pages) :

```text
====================================================================================
                            RÉPARTITION DU PROJET
====================================================================================

👤 ALIOUNE DIOP (Parties 1 & 2)                  👤 MADJIGUÈNE FALL (Parties 3 à 7)
[ Statut : ✅ TERMINÉ & POUSSÉ ]                 [ Statut : ⏳ PROCHAINE ÉTAPE ]
------------------------------------------------------------------------------------
1. Exploration & Prétraitements (Partie 1.2)     1. Extension d'architecture (Partie 3)
   - Dimensions, 561 capteurs, 6 classes            - Largeur (128, 256) & Profondeur (2-4 couches)
   - Min/Max dans [-1, 1], exclusion subject        - Nombre de paramètres & temps de calcul
2. Méthodologie Découpage (Partie 1.3)           2. Régularisation & Robustesse (Partie 4)
   - Split par volontaires vs Aléatoire             - Dropout (0.1, 0.3, 0.5), Weight Decay
   - Preuve de la supériorité (0.71% vs 2.44%)      - Batch Normalization, Early Stopping
3. Modèle de référence (Baseline - Partie 1.4)      - Sélection des 2-3 meilleures configs
   - Régression Logistique (95.99% val)          3. Évaluation Test & Confusion (Partie 5)
4. MLP 1 couche cachée (Partie 2.1)                 - Passage UNIQUE sur test.csv
   - Perte théorique ln(6) ~ 1.79                   - Matrice de confusion & analyse physique
   - Sanity check mini-lot (100% de précision)   4. Interprétabilité t-SNE / UMAP (Partie 6)
5. Activations, Optimiseurs & LR (Partie 2.2)       - Variables brutes vs Dernière couche cachée
   - ReLU vs Tanh vs Sigmoid                        - Visualisation de l'espace latent
   - SGD vs SGD+Momentum vs Adam                 5. Finalisation du Rapport & Rendu (Partie 8)
6. Variabilité multi-graines (Partie 2.3)           - Rédaction des Sections 4 à 6 (max 6 pages)
   - Répétition sur seeds 42, 123, 999              - Assemblage de l'archive Moodle finale
   - Moyenne ± écart-type                           (M2_DL_2026_DIOP_FALL.zip)
------------------------------------------------------------------------------------
Livrables fournis par Alioune :                  Livrables attendus de Madjiguène :
- Notebook Version 1 exécuté                     - Notebook Version 2 exécuté
- Module partagé src/data_loader.py              - Figures t-SNE, confusion, courbes V2
- 6 figures dans reports/figures/                - Rapport final de 6 pages (PDF)
- Rédaction sections 1 à 3 du rapport            - Archive ZIP pour Moodle
====================================================================================
```

---

## 📌 Présentation du Projet
Ce projet vise à construire et à analyser différentes architectures de réseaux de neurones multicouches (MLP) pour un problème de classification d'activités humaines à partir de signaux de capteurs (accéléromètres et gyroscopes de smartphones - HAR Dataset).

Il s'attache à comprendre l'influence de :
- La complexité de l'architecture (largeur, profondeur) ;
- Le choix des fonctions d'activation (ReLU, Tanh, Sigmoid) ;
- Les algorithmes d'optimisation (SGD, SGD Momentum, Adam) et le pas d'apprentissage ;
- Les techniques de régularisation (Dropout, BatchNorm, Weight Decay, Early Stopping) ;
- La variabilité statistique (évaluation répétée sur plusieurs graines aléatoires) ;
- L'interprétabilité des représentations latentes (t-SNE / UMAP) et l'analyse physique des erreurs de capteurs.

---

## 📁 Arborescence du Projet

```text
Projet DL/
├── .gitignore
├── requirements.txt
├── README.md
├── data/
│   └── HAR/
│       ├── train.csv                      # Données d'entraînement (7 352 lignes, 563 colonnes)
│       ├── test.csv                       # Données de test (2 947 lignes, 563 colonnes)
│       └── .gitkeep
├── src/                                   # Code Python partagé et réutilisable
│   ├── __init__.py
│   ├── data_loader.py                     # Chargement et découpage par volontaire (FAIT PAR ALIOUNE)
│   ├── train_v1_baseline.py               # Entraînement complet V1 (FAIT PAR ALIOUNE)
│   ├── experiment_split_comparison.py     # Expérience de découpage (FAIT PAR ALIOUNE)
│   ├── generate_exploratory_figures.py    # Génération des figures 01 à 04 (FAIT PAR ALIOUNE)
│   └── populate_notebook_outputs.py       # Injection des sorties notebook (FAIT PAR ALIOUNE)
├── notebooks/
│   ├── version1_alioune/                  # ✅ Version 1 : Baseline & Réseau 1 couche (ALIOUNE)
│   │   ├── README.md
│   │   └── v1_baseline_mlp_alioune.ipynb
│   └── version2_madjiguene/               # ⏳ Version 2 : Réseau profond & Interprétabilité (MADJIGUÈNE)
│       ├── README.md                      # Guide pas à pas détaillé pour Madjiguène
│       └── v2_deep_mlp_madjiguene.ipynb   # Notebook prêt à être complété
├── reports/                               # Rapport scientifique (max 6 pages)
│   ├── figures/                           # Graphiques et visualisations (01 à 06 déjà générées)
│   ├── README.md
│   ├── rapport_partie_alioune.md          # Sections 1, 2 et 3 rédigées par Alioune
│   └── M2_DL_2026_DIOP_FALL.pdf           # Rapport PDF final à compiler
└── deliverables/                          # Archives ZIP
    ├── Projet_DL_Alioune_DIOP.zip         # Travail d'Alioune zippé
    └── M2_DL_2026_DIOP_FALL.zip           # Archive finale finale pour Moodle
```

---

## 🚀 Guide de Démarrage pour Madjiguène

1. **Cloner le projet :**
   ```bash
   git clone https://github.com/aliounedjamil/Projet-DL-HAR.git
   cd Projet-DL-HAR
   ```

2. **Installer les dépendances :**
   ```bash
   pip install -r requirements.txt
   ```

3. **Où travailler :**
   - Ouvre le dossier `notebooks/version2_madjiguene/` et lis le fichier [notebooks/version2_madjiguene/README.md](notebooks/version2_madjiguene/README.md).
   - Complète le notebook `v2_deep_mlp_madjiguene.ipynb` en important directement les fonctions de `src/data_loader.py`.
   - Utilise le document `reports/rapport_partie_alioune.md` pour fusionner tes sections 4, 5 et 6 dans le rapport final !
