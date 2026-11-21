import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from data_loader import load_har_data, split_by_subject, split_random, prepare_features_and_labels

# 1. Chargement des données brutes
train_df, test_df = load_har_data("data/HAR")

# 2. Préparation du jeu de test final
X_train_full, y_train_full, le, _ = prepare_features_and_labels(train_df)
X_test, y_test, _, _ = prepare_features_and_labels(test_df, label_encoder=le)

# ==========================================================
# EXPÉRIENCE 1 : Découpage aléatoire (lignes mélangées)
# ==========================================================
train_rand, val_rand = split_random(train_df, val_ratio=0.2, random_state=42)
X_train_r, y_train_r, _, _ = prepare_features_and_labels(train_rand, label_encoder=le)
X_val_r, y_val_r, _, _ = prepare_features_and_labels(val_rand, label_encoder=le)

# Entraînement de la Régression Logistique (modèle simple de référence)
model_rand = LogisticRegression(max_iter=1000, random_state=42)
model_rand.fit(X_train_r, y_train_r)

acc_val_r = accuracy_score(y_val_r, model_rand.predict(X_val_r))
acc_test_r = accuracy_score(y_test, model_rand.predict(X_test))

# ==========================================================
# EXPÉRIENCE 2 : Découpage par volontaires entiers (Group Split)
# ==========================================================
# On réserve 4 volontaires complets pour la validation : sujets 1, 3, 5, 7
val_subjects = [1, 3, 5, 7]
train_subj, val_subj = split_by_subject(train_df, val_subjects=val_subjects)
X_train_s, y_train_s, _, _ = prepare_features_and_labels(train_subj, label_encoder=le)
X_val_s, y_val_s, _, _ = prepare_features_and_labels(val_subj, label_encoder=le)

model_subj = LogisticRegression(max_iter=1000, random_state=42)
model_subj.fit(X_train_s, y_train_s)

acc_val_s = accuracy_score(y_val_s, model_subj.predict(X_val_s))
acc_test_s = accuracy_score(y_test, model_subj.predict(X_test))

# ==========================================================
# AFFICHAGE DES RÉSULTATS COMPARATIFS
# ==========================================================
print("=" * 65)
print("COMPARAISON DES MÉTHODOLOGIES DE DÉCOUPAGE (Régression Logistique)")
print("=" * 65)
print(f"Découpage Aléatoire (Random Split 80/20) :")
print(f"  - Validation Accuracy : {acc_val_r * 100:.2f} %")
print(f"  - Test Accuracy       : {acc_test_r * 100:.2f} %")
print(f"  - Écart |Val - Test|  : {abs(acc_val_r - acc_test_r) * 100:.2f} % (Estimation trop optimiste !)")

print("-" * 65)
print(f"Découpage par Volontaires (Sujets {val_subjects}) :")
print(f"  - Validation Accuracy : {acc_val_s * 100:.2f} %")
print(f"  - Test Accuracy       : {acc_test_s * 100:.2f} %")
print(f"  - Écart |Val - Test|  : {abs(acc_val_s - acc_test_s) * 100:.2f} % (Très proche de la réalité)")
print("=" * 65)

