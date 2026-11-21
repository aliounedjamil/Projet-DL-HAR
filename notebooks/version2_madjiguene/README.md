# 👤 Guide de Travail : Madjiguène FALL (Notebook Version 2)

Bienvenue Madjiguène !  
Alioune a terminé l'ensemble des **Parties 1 et 2** (exploration, découpage par volontaires, baseline régression logistique, réseau simple à 1 couche cachée, comparaison activations/optimiseurs, étude de variabilité multi-graines).

Tout le socle est prêt et importable. Voici ta feuille de route pas à pas pour réaliser la **Version 2** et finaliser le projet.

---

## 🛠️ Ce qui est déjà prêt pour toi :
1. **Les données et DataLoaders** :
   Dans ton notebook, tu peux importer directement :
   ```python
   import sys
   sys.path.append('../../src')
   from data_loader import load_har_data, split_by_subject, prepare_features_and_labels, create_dataloaders
   ```
2. **Les fonctions d'entraînement de base** :
   Tu peux t'inspirer de `src/train_v1_baseline.py` pour réutiliser les fonctions `train_epoch` et `evaluate`.
3. **La rédaction des 3 premières pages du rapport** :
   Consulte `reports/rapport_partie_alioune.md` pour voir ce qu'Alioune a rédigé et y ajouter directement tes sections 4, 5 et 6.

---

## 🎯 Tes Tâches Étape par Étape :

### 1. Partie 3 : Extension de l'architecture
* **Largeur (Section 3.1)** :
  * Augmente progressivement la taille de la couche cachée : 128 neurones, puis 256 neurones.
  * Analyse l'impact sur la performance en validation et sur l'overfitting.
  * Calcule et affiche le nombre de paramètres de chaque modèle (`sum(p.numel() for p in model.parameters())`).
* **Profondeur (Section 3.2)** :
  * Implémente des réseaux plus profonds (2, 3 et 4 couches cachées).
  * Teste différentes combinaisons d'activations dans les couches intermédiaires.
  * Compare l'accuracy, le macro F1-score, le nombre de paramètres et le temps d'entraînement.

### 2. Partie 4 : Régularisation et robustesse
* Si tu observes du surapprentissage (écart grandissant entre train loss et val loss), teste :
  * **Dropout** : teste différentes probabilités $p \in \{0.1, 0.3, 0.5\}$ après les couches cachées.
  * **Weight Decay (L2)** : via l'optimiseur (ex: `weight_decay=1e-4`).
  * **Batch Normalization** : ajoute `nn.BatchNorm1d` avant ou après les activations.
  * **Early Stopping** : arrêt automatique si la perte de validation ne s'améliore plus après $K$ époques.
* Sélectionne les **2 ou 3 configurations les plus performantes** sur l'ensemble de validation.

### 3. Partie 5 : Visualisation et Évaluation Finale
* **Courbes d'apprentissage (Section 5.1)** :
  * Trace l'évolution de la loss et du F1-score/accuracy sur le train et la validation au cours des époques.
* **Évaluation finale sur le test (Section 5.2 - RÈGLE D'OR : 1 seule fois !)** :
  * Évalue les 2 ou 3 meilleures configurations ainsi que la baseline de référence sur le jeu de test final (`test_loader`).
* **Matrice de confusion** :
  * Calcule et affiche la matrice de confusion sur le test avec `seaborn.heatmap`.
  * **Analyse physique des erreurs (indispensable pour la soutenance)** :
    * Pourquoi *SITTING* et *STANDING* sont-ils souvent confondus ? *(Les deux sont des activités statiques où le capteur ne mesure principalement que la gravité terrestre).*
    * Pourquoi *WALKING_UPSTAIRS* et *WALKING_DOWNSTAIRS* peuvent-ils se confondre ? *(Même cadence périodique de pas, légères différences sur l'axe vertical).*

### 4. Partie 6 : Interprétabilité et Réduction de dimension
* Applique **t-SNE** (`sklearn.manifold.TSNE`) ou **UMAP** (`import umap`) :
  1. Sur les **variables d'entrée brutes** (les 561 capteurs).
  2. Sur les **activations de la dernière couche cachée** de ton meilleur réseau.
* Compare visuellement les deux projections :
  * Montre comment le réseau a appris à séparer les classes dans son espace latent par rapport à l'espace initial.
  * Vérifie si les classes qui se chevauchent dans la projection correspondent à celles confondues dans la matrice de confusion.

### 5. Finalisation du Rapport & Rendu (Partie 8)
* Rédige les sections 4, 5 et 6 du rapport en t'appuyant sur `reports/rapport_partie_alioune.md`.
* Compile le rapport final en PDF (limite stricte : **max 6 pages**) nommé `reports/M2_DL_2026_DIOP_FALL.pdf`.
* Crée l'archive finale `M2_DL_2026_DIOP_FALL.zip` prête à soumettre sur Moodle !
