# M2 MALIA — Deep Learning : Projet HAR (Human Activity Recognition)

**Binôme :** Alioune DIOP · Madjiguène FALL
**Enseignant :** Walid Bechkit — Université Lumière Lyon 2 — 2026–2027

Étude progressive de réseaux de neurones multicouches (MLP) pour reconnaître 6 activités à partir des 561 variables
calculées sur les capteurs (accéléromètre, gyroscope) d'un smartphone.

## Répartition

| Alioune DIOP — Version 1 | Madjiguène FALL — Version 2 |
|---|---|
| Exploration et prétraitements (1.2) | Largeur et profondeur (3) |
| Découpage par volontaire vs aléatoire (1.3) | Régularisation : Dropout, L2, BatchNorm, early stopping (4) |
| Référence : régression logistique (1.4) | Courbes d'apprentissage, évaluation finale, confusion (5) |
| MLP 1 couche : contrôles, réglage du pas, activations, optimiseurs (2.1–2.2) | t-SNE couche cachée vs entrées (6) |
| Variabilité sur 3 graines (2.3) et test unique de la V1 | Ablation par famille de capteurs (7) |

Rapport : rédigé à deux.

## Arborescence

```
├── data/HAR/                      # train.csv, test.csv (Kaggle) — non versionnés
├── notebooks/
│   ├── version1_alioune/v1_baseline_mlp_alioune.ipynb
│   └── version2_madjiguene/v2_deep_mlp_madjiguene.ipynb
├── reports/
│   ├── figures/                   # figures produites par les notebooks
│   ├── results/                   # v1_results.json, v2_results.json : chiffres produits par les notebooks
│   └── M2_DL_2026_DIOP_FALL.pdf   # rapport (6 pages max)
├── src/
│   ├── data_loader.py             # chargement, découpage par volontaire, DataLoaders (utilisé par la V2)
│   ├── generate_exploratory_figures.py
│   └── create_final_submission_zip.py
└── requirements.txt
```

## Exécution

```bash
pip install -r requirements.txt
# placer train.csv et test.csv dans data/HAR/
jupyter notebook
```

1. `notebooks/version1_alioune/v1_baseline_mlp_alioune.ipynb` — *Restart & Run All* (≈ 10 min sur CPU)
2. `notebooks/version2_madjiguene/v2_deep_mlp_madjiguene.ipynb` — *Restart & Run All* (≈ 15–20 min sur CPU)

Chaque notebook enregistre ses figures dans `reports/figures/` et ses chiffres dans `reports/results/`.
Toutes les sorties affichées dans les notebooks sont produites par leur exécution.

**Méthodologie.** Validation = volontaires 1, 3, 5, 7 (aucune fenêtre d'un même volontaire des deux côtés).
Tous les choix sont faits sur la validation ; `test.csv` n'est évalué qu'une fois, en dernière section de chaque
notebook, sur les configurations figées. Les comparaisons importantes sont répétées sur 3 graines (42, 123, 999).
