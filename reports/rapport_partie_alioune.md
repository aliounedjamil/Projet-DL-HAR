# M2 MALIA - Deep Learning
## Projet : Étude progressive des réseaux de neurones sur le jeu de données HAR
**Enseignant :** Walid Bechkit — **Année universitaire :** 2026–2027  
**Auteurs :** Alioune DIOP & Madjiguène FALL

---

# PARTIE RÉDIGÉE PAR ALIOUNE DIOP (Sections 1 à 3)

## 1. Introduction et Problématique
La reconnaissance d'activités humaines (*Human Activity Recognition* - HAR) à partir de dispositifs portables constitue une tâche fondamentale pour l'assistance médicale, le suivi sportif et les interfaces homme-machine. Ce projet vise à concevoir, évaluer et analyser de manière progressive des architectures de réseaux de neurones multicouches (MLP) sur des données issues d'accéléromètres et de gyroscopes tri-axiaux de smartphones.

L'objectif méthodologique central n'est pas uniquement d'atteindre une précision brute maximale, mais d'étudier pas à pas l'impact de chaque choix de conception (largeur, profondeur, activations, algorithmes d'optimisation, régularisations) à travers une démarche scientifique rigoureuse : définition d'un modèle de référence simple, mise en place d'un protocole de validation sans fuite de données, analyse de la variabilité statistique par répétition de graines aléatoires, et sanctuarisation de l'ensemble de test pour une évaluation finale unique.

---

## 2. Présentation du Jeu de Données et Démarche Expérimentale

### 2.1 Exploration et Caractéristiques du Jeu de Données
Le jeu de données HAR est composé d'enregistrements capturés par un smartphone fixé à la taille de 30 volontaires réalisant 6 activités distinctes :
1. `WALKING` (Marche)
2. `WALKING_UPSTAIRS` (Montée d'escaliers)
3. `WALKING_DOWNSTAIRS` (Descente d'escaliers)
4. `SITTING` (Position assise)
5. `STANDING` (Station debout)
6. `LAYING` (Position allongée)

Le jeu d'entraînement (`train.csv`) regroupe **7 352 observations** réparties sur 21 volontaires, tandis que le jeu de test (`test.csv`) compte **2 947 observations** réparties sur 9 volontaires. Aucun volontaire n'est présent simultanément dans le train et dans le test, ce qui garantit une évaluation en condition réelle sur de nouveaux individus.

Chaque observation correspond à une fenêtre temporelle de 2,56 secondes (échantillonnée à 50 Hz), décrite par **561 caractéristiques** calculées dans les domaines temporel et fréquentiel (moyennes, écarts-types, énergies, bandes spectrales, etc.). Les distributions des classes sont équilibrées (entre 13,4% et 19,1% par classe).

### 2.2 Prétraitements et Choix de Conception
- **Mise à l'échelle :** L'analyse des données montre que le minimum global de toutes les variables est de **-1.0000** et le maximum est de **+1.0000**. Les auteurs du dataset ont déjà normalisé toutes les caractéristiques dans l'intervalle $[-1, 1]$. Aucune standardisation supplémentaire n'est requise.
- **Exclusion de la colonne `subject` :** La variable `subject` représente l'identifiant du volontaire. Elle ne doit en aucun cas être injectée en entrée du modèle : son utilisation introduirait un biais d'apprentissage majeur (*data leakage*), le modèle apprenant à identifier la personne plutôt que le mouvement, ce qui détruirait sa capacité de généralisation sur les volontaires inconnus du jeu de test.
- **Encodage de la cible :** Les étiquettes textuelles sont encodées en entiers de $0$ à $5$ afin d'alimenter la fonction de perte multiclasse `nn.CrossEntropyLoss`.

### 2.3 Méthodologie de Validation : Découpage par Volontaire vs Découpage Aléatoire
Dans le protocole d'acquisition du jeu HAR, les fenêtres temporelles consécutives d'un même individu sont capturées avec **50% de chevauchement**. Deux fenêtres adjacentes présentent donc une très forte autocorrélation.

Pour démontrer l'importance du découpage, nous avons entraîné une régression logistique selon deux stratégies :
1. **Découpage aléatoire (80/20) :** Les lignes sont mélangées au hasard.
2. **Découpage par volontaires (*GroupSplit*) :** Nous réservons 4 volontaires entiers (sujets 1, 3, 5, 7, soit 1 298 exemples) pour la validation.

| Stratégie de Découpage | Validation Accuracy | Test Accuracy | Écart \|Val - Test\| |
| :--- | :---: | :---: | :---: |
| **Aléatoire (Random Split)** | 98,57 % | 96,13 % | 2,44 % |
| **Par Volontaires (GroupSplit)** | **95,99 %** | **95,28 %** | **0,71 %** |

**Conclusion méthodologique :** Le découpage aléatoire est artificiellement optimiste (+2,5 points au-dessus de la réalité) en raison de la fuite d'information entre fenêtres chevauchantes. Le découpage par volontaire reflète fidèlement la performance sur le jeu de test final (écart résiduel de seulement 0,71%). Il est donc retenu pour l'ensemble des expérimentations.

---

## 3. Première Architecture MLP et Résultats (Version 1)

### 3.1 Modèle de Référence (Baseline)
La régression logistique multinomiale (équivalente à un réseau à 0 couche cachée) atteint une précision de **95,99%** en validation et **95,28%** en test. Cette performance servira de référence minimale pour évaluer l'apport des réseaux de neurones.

### 3.2 Réseau à Une Couche Cachée & Contrôles Préalables
Nous implémentons un MLP comportant 561 entrées, 1 couche cachée de 64 neurones, et 6 neurones de sortie, optimisé avec `CrossEntropyLoss`. Deux contrôles indispensables ont été validés :
1. **Perte initiale théorique :** Pour des poids aléatoires et 6 classes équiprobables, la perte théorique attendue est $\mathcal{L}_0 = -\ln(1/6) = \ln(6) \approx 1{,}7918$. La perte observée sur notre premier lot est de **$1{,}7592$**, confirmant une initialisation saine.
2. **Sanity Check (Surapprentissage d'un mini-lot) :** Sur un sous-ensemble de 10 exemples, le réseau converge vers une perte de **$0{,}0029$** et une précision de **100,0%** en moins de 50 itérations, attestant de l'absence de bug dans le graphe de calcul et les gradients.

### 3.3 Étude Comparative des Fonctions d'Activation et Optimiseurs

| Fonction d'Activation | Optimiseur | Learning Rate | Val Loss | Val Accuracy | Val Macro-F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ReLU** | Adam | 0.001 | 0.1346 | 94,84 % | 94,61 % |
| **Tanh** | Adam | 0.001 | 0.1333 | **95,38 %** | **95,19 %** |
| **Sigmoid** | Adam | 0.001 | 0.1386 | 94,45 % | 94,22 % |
| ReLU | SGD classique | 0.05 | 0.2107 | 91,45 % | 91,12 % |
| ReLU | SGD + Momentum (0.9) | 0.01 | 0.1246 | **95,53 %** | **95,34 %** |

**Observations :**
- L'activation **Tanh** surpasse légèrement ReLU et Sigmoid sur cette architecture à 1 couche. Les données d'entrée étant centrées et normalisées dans $[-1, 1]$, la symétrie de la tangente hyperbolique autour de zéro favorise une convergence harmonieuse des gradients.
- L'optimiseur **SGD avec Momentum** atteint les meilleures performances (Val Acc = 95,53%), dépassant SGD classique (+4,08%) et rivalisant avec Adam, grâce à une bonne inertie face aux oscillations dans les vallées de gradient.
- L'accuracy et le Macro-F1 présentent une concordance étroite (écart < 0,2%), ce qui est cohérent avec l'équilibre des classes mesuré en Section 2.1.

### 3.4 Analyse de la Variabilité Statistique (Partie 2.3)
Pour déterminer si les écarts observés sont significatifs, nous avons réitéré les entraînements sur 3 graines aléatoires indépendantes (`seeds = [42, 123, 999]`) :

| Configuration | Graines testées | Scores individuels | Moyenne $\pm$ Écart-type |
| :--- | :---: | :---: | :---: |
| **ReLU + Adam** | 42, 123, 999 | 94,84% / 94,84% / 95,22% | **94,97 % $\pm$ 0,18 %** |
| **Tanh + Adam** | 42, 123, 999 | 95,38% / 95,61% / 95,61% | **95,53 % $\pm$ 0,11 %** |

L'écart entre les deux moyennes est de **0,56%**, ce qui est supérieur aux écarts-types mesurés (0,18% et 0,11%). L'avantage de l'activation Tanh sur ce réseau simple peut donc être qualifié de statistiquement significatif à ce stade.

---
*(La suite du rapport : Section 4 - Architectures profondes et régularisation, Section 5 - Évaluation finale sur test et matrice de confusion, et Section 6 - Interprétabilité t-SNE / UMAP, est rédigée par Madjiguène FALL)*
