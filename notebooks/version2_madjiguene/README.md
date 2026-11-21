# Notebook Version 2 : Madjiguène FALL

Ce dossier contient les travaux et le notebook réalisés par **Madjiguène FALL** :

### Objectifs :
1. **Extension de l'architecture** :
   - Élargissement de la couche cachée (128, 256 neurones) et étude de l'overfitting.
   - Réseaux plus profonds (2 à 4 couches cachées), combinaisons d'activations, calcul du nombre de paramètres et temps de calcul.
2. **Régularisation et robustesse** :
   - Dropout (p = 0.1, 0.3, 0.5), Weight Decay (L2), Batch Normalization, Early Stopping.
   - Sélection des 2 ou 3 meilleures configurations sur l'ensemble de validation.
3. **Visualisation et analyse des performances** :
   - Courbes d'apprentissage comparatives (Loss et F1-score/Accuracy).
   - Évaluation **unique** sur le jeu de test final (`test.csv`).
   - Matrice de confusion sur le jeu de test et discussion physique des erreurs de capteurs.
4. **Interprétabilité et réduction de dimension** :
   - Analyse t-SNE et UMAP comparant les variables d'entrée brutes aux représentations de la dernière couche cachée.
   - Visualisation de la séparabilité de l'espace latent et corrélation avec la matrice de confusion.
5. **Ouvertures (Bonus)** :
   - Automatisation de la recherche d'hyperparamètres ou analyse d'importance des variables.
