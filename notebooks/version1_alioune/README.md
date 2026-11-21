# Notebook Version 1 : Alioune DIOP

Ce dossier contient les travaux et le notebook réalisés par **Alioune DIOP** :

### Objectifs :
1. **Prise en main exploratoire des données** :
   - Dimensions, analyse des 561 variables, des volontaires (`subject`) et des 6 classes d'activité.
   - Statistiques descriptives et discussion sur la mise à l'échelle des données.
2. **Méthodologie de validation rigoureuse** :
   - Découpage par volontaires entiers (*GroupSplit*) vs découpage aléatoire.
   - Justification physique (chevauchement à 50% des fenêtres temporelles consécutives d'un même individu).
3. **Référence simple (Baseline)** :
   - Régression logistique multiclasse (`scikit-learn` / PyTorch 0 couche cachée).
4. **Première architecture de réseau de neurones** :
   - Réseau MLP à une seule couche cachée (ex. 64 neurones) avec `CrossEntropyLoss`.
   - Vérification de la perte initiale théorique ($\approx \ln(6) \approx 1{,}79$).
   - Test de surapprentissage d'un mini-lot (*sanity check*).
   - Étude comparative des fonctions d'activation (ReLU, Sigmoid, Tanh).
   - Étude comparative des optimiseurs (SGD, SGD avec momentum, Adam) et réglage du learning rate.
   - Suivi des courbes d'apprentissage (Loss, Accuracy, Macro F1-score).
5. **Étude de la variabilité des résultats** :
   - Entraînements répétés sur au moins 3 graines aléatoires (moyennes $\pm$ écarts-types).
