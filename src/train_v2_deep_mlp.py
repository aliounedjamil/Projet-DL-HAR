"""
Script complet d'entraînement et d'analyse pour la Version 2 (Parties 3 à 7)
Binôme : Alioune DIOP & Madjiguène FALL (Réalisé pour Madjiguène)
Enseignant : Walid Bechkit — M2 MALIA Deep Learning

Ce script exécute :
- Partie 3 : Extension de l'architecture (largeur 64, 128, 256 ; profondeur 1 à 4 couches)
- Partie 4 : Régularisation (Dropout, Weight Decay, BatchNorm, Early Stopping)
- Partie 5 : Visualisation des courbes d'apprentissage et Évaluation finale sur Test (Matrice de confusion)
- Partie 6 : Interprétabilité t-SNE (features brutes vs espace latent)
- Partie 7 : Bonus / Ouverture (Ablation study par type de capteur)
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import load_har_data, split_by_subject, prepare_features_and_labels, create_dataloaders

# Configuration
os.makedirs("reports/figures", exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ==============================================================================
# MODÈLE PROFOND MODULAIRE
# ==============================================================================
class DeepMLP(nn.Module):
    def __init__(self, input_dim=561, hidden_dims=[128, 64], num_classes=6, 
                 activation='relu', dropout_p=0.0, use_bn=False):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(h_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            prev_dim = h_dim
            
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)
        
    def forward(self, x):
        feat = self.features(x)
        out = self.classifier(feat)
        return out
    
    def extract_features(self, x):
        """Extrait la représentation latente de la dernière couche cachée"""
        return self.features(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Fonctions d'entraînement et d'évaluation
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x_b, y_b in dataloader:
        optimizer.zero_grad()
        out = model(x_b)
        loss = criterion(out, y_b)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x_b.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y_b).sum().item()
        total += x_b.size(0)
    return total_loss / total, correct / total

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_b, y_b in dataloader:
            out = model(x_b)
            loss = criterion(out, y_b)
            total_loss += loss.item() * x_b.size(0)
            all_preds.extend(out.argmax(dim=1).cpu().numpy())
            all_targets.extend(y_b.cpu().numpy())
            
    total = len(all_targets)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    return total_loss / total, acc, macro_f1, all_preds, all_targets

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=35, patience=7):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    start_time = time.time()
    for ep in range(epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, vl_f1, _, _ = evaluate(model, val_loader, criterion)
        
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)
        history['val_f1'].append(vl_f1)
        
        # Early Stopping sur la val loss
        if vl_loss < best_loss:
            best_loss = vl_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.time() - start_time
    return model, history, elapsed

# ==============================================================================
# SCRIPT PRINCIPAL
# ==============================================================================
def main():
    print("=" * 75)
    print("M2 MALIA DEEP LEARNING - EXÉCUTION PARTIES 3 À 7 (MADJIGUÈNE FALL)")
    print("=" * 75)
    
    # 1. Chargement des données
    print("\n[1] Chargement des données et création des DataLoaders...")
    train_df, test_df = load_har_data("data/HAR")
    val_subjects = [1, 3, 5, 7]
    train_split, val_split = split_by_subject(train_df, val_subjects=val_subjects)
    
    X_tr, y_tr, le, feature_cols = prepare_features_and_labels(train_split)
    X_val, y_val, _, _ = prepare_features_and_labels(val_split, label_encoder=le)
    X_test, y_test, _, _ = prepare_features_and_labels(test_df, label_encoder=le)
    
    batch_size = 64
    train_loader, val_loader = create_dataloaders(X_tr, y_tr, X_val, y_val, batch_size=batch_size)
    _, test_loader = create_dataloaders(X_tr, y_tr, X_test, y_test, batch_size=batch_size)
    
    print(f"   Train : {len(X_tr)} obs | Val : {len(X_val)} obs | Test final : {len(X_test)} obs")
    print(f"   Classes : {list(le.classes_)}")
    
    # ==========================================================================
    # PARTIE 3 : EXTENSION DE L'ARCHITECTURE (Largeur & Profondeur)
    # ==========================================================================
    print("\n[2] Partie 3 : Étude de l'architecture (Largeur et Profondeur)...")
    arch_configs = [
        ("1 couche (64) [V1]", [64], 'relu'),
        ("1 couche (128)", [128], 'relu'),
        ("1 couche (256)", [256], 'relu'),
        ("2 couches [128, 64]", [128, 64], 'relu'),
        ("3 couches [256, 128, 64] (ReLU)", [256, 128, 64], 'relu'),
        ("3 couches [256, 128, 64] (Tanh)", [256, 128, 64], 'tanh'),
        ("4 couches [256, 128, 64, 32]", [256, 128, 64, 32], 'relu')
    ]
    
    arch_results = []
    crit = nn.CrossEntropyLoss()
    
    for name, dims, act in arch_configs:
        set_seed(42)
        model = DeepMLP(hidden_dims=dims, activation=act)
        opt = optim.Adam(model.parameters(), lr=0.001)
        
        model, hist, duration = train_model(model, train_loader, val_loader, crit, opt, epochs=25, patience=6)
        n_params = count_parameters(model)
        
        final_tr_loss = hist['train_loss'][-1]
        final_vl_loss = hist['val_loss'][-1]
        best_val_acc = max(hist['val_acc']) * 100
        best_val_f1 = max(hist['val_f1']) * 100
        overfit_gap = (hist['train_acc'][-1] - hist['val_acc'][-1]) * 100
        
        arch_results.append({
            'Architecture': name,
            'Paramètres': n_params,
            'Temps (s)': round(duration, 2),
            'Train Loss': round(final_tr_loss, 4),
            'Val Loss': round(final_vl_loss, 4),
            'Val Acc (%)': round(best_val_acc, 2),
            'Val F1 (%)': round(best_val_f1, 2),
            'Écart Overfit (%)': round(overfit_gap, 2)
        })
        print(f"   -> {name:32} | Params: {n_params:6,d} | Val Acc: {best_val_acc:.2f}% | Val F1: {best_val_f1:.2f}% | Temps: {duration:.1f}s")
        
    df_arch = pd.DataFrame(arch_results)
    
    # Figure 07 : Comparaison Largeur & Profondeur
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=df_arch, y='Architecture', x='Val Acc (%)', ax=axes[0], palette='Blues_r')
    axes[0].set_title("Validation Accuracy par Architecture", fontweight='bold')
    axes[0].set_xlim(90, 98)
    for i, v in enumerate(df_arch['Val Acc (%)']):
        axes[0].text(v + 0.1, i, f"{v:.2f}%", va='center', fontsize=9, fontweight='bold')
        
    sns.barplot(data=df_arch, y='Architecture', x='Paramètres', ax=axes[1], palette='Purples_r')
    axes[1].set_title("Nombre de Paramètres par Architecture", fontweight='bold')
    plt.tight_layout()
    plt.savefig("reports/figures/07_comparaison_largeur_profondeur.png", dpi=300)
    plt.close()
    print("   -> Figure sauvegardée : reports/figures/07_comparaison_largeur_profondeur.png")
    
    # ==========================================================================
    # PARTIE 4 : RÉGULARISATION ET ROBUSTESSE
    # ==========================================================================
    print("\n[3] Partie 4 : Régularisation et Robustesse (sur l'architecture 3 couches)...")
    reg_configs = [
        ("Sans régularisation", 0.0, False, 0.0),
        ("Dropout (p=0.1)", 0.1, False, 0.0),
        ("Dropout (p=0.3)", 0.3, False, 0.0),
        ("Dropout (p=0.5)", 0.5, False, 0.0),
        ("Weight Decay (L2 = 1e-4)", 0.0, False, 1e-4),
        ("Batch Normalization", 0.0, True, 0.0),
        ("BatchNorm + Dropout 0.3", 0.3, True, 0.0),
        ("BatchNorm + Dropout 0.3 + L2 (1e-4)", 0.3, True, 1e-4)
    ]
    
    reg_results = []
    reg_histories = {}
    best_overall_model = None
    best_overall_val_acc = 0.0
    best_overall_name = ""
    best_overall_history = None
    
    for label, drop_p, use_bn, l2 in reg_configs:
        set_seed(42)
        model = DeepMLP(hidden_dims=[256, 128, 64], activation='relu', dropout_p=drop_p, use_bn=use_bn)
        opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=l2)
        
        model, hist, duration = train_model(model, train_loader, val_loader, crit, opt, epochs=30, patience=7)
        reg_histories[label] = hist
        
        best_acc = max(hist['val_acc']) * 100
        best_f1 = max(hist['val_f1']) * 100
        final_vl = min(hist['val_loss'])
        
        reg_results.append({
            'Régularisation': label,
            'Val Loss': round(final_vl, 4),
            'Val Acc (%)': round(best_acc, 2),
            'Val F1 (%)': round(best_f1, 2)
        })
        print(f"   -> {label:36} | Val Loss: {final_vl:.4f} | Val Acc: {best_acc:.2f}% | Val F1: {best_f1:.2f}%")
        
        if best_acc > best_overall_val_acc:
            best_overall_val_acc = best_acc
            best_overall_model = model
            best_overall_name = label
            best_overall_history = hist
            
    df_reg = pd.DataFrame(reg_results)
    
    # Figure 08 : Comparaison Régularisation
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=df_reg, y='Régularisation', x='Val Acc (%)', palette='crest')
    plt.title("Impact des Techniques de Régularisation (Architecture 3 couches)", fontweight='bold')
    plt.xlim(92, 98)
    for i, v in enumerate(df_reg['Val Acc (%)']):
        ax.text(v + 0.08, i, f"{v:.2f}%", va='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig("reports/figures/08_comparaison_regularisation.png", dpi=300)
    plt.close()
    print("   -> Figure sauvegardée : reports/figures/08_comparaison_regularisation.png")
    
    # ==========================================================================
    # PARTIE 5 : COURBES D'APPRENTISSAGE & ÉVALUATION FINALE (TEST)
    # ==========================================================================
    print("\n[4] Partie 5 : Courbes d'apprentissage & Évaluation finale sur Test...")
    
    # Figure 09 : Courbes d'apprentissage du meilleur modèle
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    epochs_range = range(1, len(best_overall_history['train_loss']) + 1)
    
    axes[0].plot(epochs_range, best_overall_history['train_loss'], label='Train Loss', color='royalblue', lw=2)
    axes[0].plot(epochs_range, best_overall_history['val_loss'], label='Val Loss', color='crimson', lw=2, linestyle='--')
    axes[0].set_title("Courbe de Perte (Loss) - Meilleur Modèle", fontweight='bold')
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    
    axes[1].plot(epochs_range, [x*100 for x in best_overall_history['train_acc']], label='Train Accuracy', color='royalblue', lw=2)
    axes[1].plot(epochs_range, [x*100 for x in best_overall_history['val_acc']], label='Val Accuracy', color='crimson', lw=2, linestyle='--')
    axes[1].set_title("Courbe de Précision (Accuracy) - Meilleur Modèle", fontweight='bold')
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("Précision (%)")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig("reports/figures/09_courbes_apprentissage_meilleur_modele.png", dpi=300)
    plt.close()
    print("   -> Figure sauvegardée : reports/figures/09_courbes_apprentissage_meilleur_modele.png")
    
    # ÉVALUATION TEST FINALE (Passage UNIQUE sur test.csv)
    print("\n   [Évaluation Test Finale - Passage Unique]")
    # 1. Baseline Régression Logistique (Partie 1)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_tr, y_tr)
    acc_val_lr = accuracy_score(y_val, lr.predict(X_val)) * 100
    acc_test_lr = accuracy_score(y_test, lr.predict(X_test)) * 100
    f1_test_lr = f1_score(y_test, lr.predict(X_test), average='macro') * 100
    
    # 2. MLP 1 couche (Partie 2)
    mlp_v1 = DeepMLP(hidden_dims=[64], activation='relu')
    set_seed(42)
    mlp_v1, hist_v1, _ = train_model(mlp_v1, train_loader, val_loader, crit, optim.Adam(mlp_v1.parameters(), lr=0.001), epochs=20)
    _, acc_val_v1, _, _, _ = evaluate(mlp_v1, val_loader, crit)
    _, acc_test_v1, f1_test_v1, _, _ = evaluate(mlp_v1, test_loader, crit)
    acc_val_v1 *= 100
    acc_test_v1 *= 100
    f1_test_v1 *= 100
    
    # 3. Meilleur MLP Profond Régularisé (Partie 4)
    _, acc_val_deep, _, _, _ = evaluate(best_overall_model, val_loader, crit)
    _, acc_test_deep, f1_test_deep, preds_test, targets_test = evaluate(best_overall_model, test_loader, crit)
    acc_val_deep *= 100
    acc_test_deep *= 100
    f1_test_deep *= 100
    
    print("\n" + "=" * 75)
    print("TABLEAU COMPARATIF FINAL (RÉFÉRENCE & ARCHITECTURES)")
    print("=" * 75)
    print(f"{'Modèle':38} | {'Val Acc (%)':11} | {'Test Acc (%)':12} | {'Test F1 (%)':11} | {'Écart |Val-Test|':15}")
    print("-" * 75)
    print(f"{'1. Régression Logistique (Baseline)':38} | {acc_val_lr:11.2f} | {acc_test_lr:12.2f} | {f1_test_lr:11.2f} | {abs(acc_val_lr-acc_test_lr):15.2f}")
    print(f"{'2. Simple MLP 1 couche [64] (V1)':38} | {acc_val_v1:11.2f} | {acc_test_v1:12.2f} | {f1_test_v1:11.2f} | {abs(acc_val_v1-acc_test_v1):15.2f}")
    print(f"{'3. Deep MLP 3 couches + ' + best_overall_name:38} | {acc_val_deep:11.2f} | {acc_test_deep:12.2f} | {f1_test_deep:11.2f} | {abs(acc_val_deep-acc_test_deep):15.2f}")
    print("=" * 75)
    
    # Matrice de confusion sur Test
    cm = confusion_matrix(targets_test, preds_test)
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Matrice de Confusion sur le Jeu de Test ({acc_test_deep:.2f}%)", fontweight='bold')
    plt.xlabel("Activité Prédite", fontweight='bold')
    plt.ylabel("Activité Réelle", fontweight='bold')
    plt.xticks(rotation=40, ha='right')
    plt.tight_layout()
    plt.savefig("reports/figures/10_matrice_confusion_test.png", dpi=300)
    plt.close()
    print("   -> Figure sauvegardée : reports/figures/10_matrice_confusion_test.png")
    
    print("\nRapport de classification par classe sur le test :")
    print(classification_report(targets_test, preds_test, target_names=le.classes_))
    
    # ==========================================================================
    # PARTIE 6 : INTERPRÉTABILITÉ ET RÉDUCTION DE DIMENSION (t-SNE)
    # ==========================================================================
    print("\n[5] Partie 6 : Interprétabilité et Réduction de Dimension (t-SNE)...")
    # Pour t-SNE, on prend un échantillon représentatif de 1200 points du test
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test), size=1200, replace=False)
    X_sample = X_test[sample_indices]
    y_sample = y_test[sample_indices]
    
    # 1. t-SNE sur l'espace d'entrée brut (561 dimensions)
    print("   Calcul t-SNE sur les données d'entrée brutes (561 capteurs)...")
    tsne_raw = TSNE(n_components=2, perplexity=35, random_state=42, max_iter=800)
    emb_raw = tsne_raw.fit_transform(X_sample)
    
    # 2. t-SNE sur l'espace latent de la dernière couche cachée
    print("   Extraction des activations de la dernière couche cachée du réseau...")
    best_overall_model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X_sample)
        # On extrait la sortie avant le classifier linéaire
        latent_features = best_overall_model.extract_features(x_tensor).cpu().numpy()
        
    print(f"   Calcul t-SNE sur l'espace latent ({latent_features.shape[1]} dimensions)...")
    tsne_latent = TSNE(n_components=2, perplexity=35, random_state=42, max_iter=800)
    emb_latent = tsne_latent.fit_transform(latent_features)
    
    # Figure 11 : t-SNE Brut vs Latent
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    palette = sns.color_palette("tab10", 6)
    
    for i, act in enumerate(le.classes_):
        mask = (y_sample == i)
        axes[0].scatter(emb_raw[mask, 0], emb_raw[mask, 1], label=act, color=palette[i], alpha=0.7, s=25)
        axes[1].scatter(emb_latent[mask, 0], emb_latent[mask, 1], label=act, color=palette[i], alpha=0.7, s=25)
        
    axes[0].set_title("Espace d'Entrée Brut (561 variables de capteurs)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")
    axes[0].legend(loc='upper right', fontsize=8)
    
    axes[1].set_title("Espace Latent Appris (Dernière Couche Cachée)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    axes[1].legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig("reports/figures/11_tsne_brut_vs_latent.png", dpi=300)
    plt.close()
    print("   -> Figure sauvegardée : reports/figures/11_tsne_brut_vs_latent.png")
    
    # ==========================================================================
    # PARTIE 7 : BONUS / OUVERTURE (Ablation Study par groupe de capteurs)
    # ==========================================================================
    print("\n[6] Partie 7 : Bonus / Ouverture (Étude d'ablation par type de capteur)...")
    # Groupes de variables :
    # - Accélération temporelle corporelle : 'tBodyAcc'
    # - Accélération de pesanteur (gravité) : 'tGravityAcc'
    # - Gyroscope temporel : 'tBodyGyro'
    # - Domaine fréquentiel (Fourier) : 'fBody'
    sensor_groups = {
        'Tous les capteurs (561 vars)': [c for c in feature_cols],
        'Sans Gravité (pas de tGravity)': [c for c in feature_cols if 'tGravity' not in c],
        'Sans Gyroscope (Accéléromètre seul)': [c for c in feature_cols if 'Gyro' not in c],
        'Domaine Temporel seul (sans Fourier)': [c for c in feature_cols if not c.startswith('f')],
        'Domaine Fréquentiel seul (Fourier seul)': [c for c in feature_cols if c.startswith('f')]
    }
    
    ablation_results = []
    for g_name, cols in sensor_groups.items():
        col_indices = [feature_cols.index(c) for c in cols]
        X_tr_sub = X_tr[:, col_indices]
        X_val_sub = X_val[:, col_indices]
        
        tr_sub_loader, val_sub_loader = create_dataloaders(X_tr_sub, y_tr, X_val_sub, y_val, batch_size=64)
        set_seed(42)
        m_abl = DeepMLP(input_dim=len(cols), hidden_dims=[128, 64], activation='relu')
        m_abl, hist_abl, _ = train_model(m_abl, tr_sub_loader, val_sub_loader, crit, optim.Adam(m_abl.parameters(), lr=0.001), epochs=20)
        best_abl_acc = max(hist_abl['val_acc']) * 100
        
        ablation_results.append({
            'Groupe de variables': g_name,
            'Nb Variables': len(cols),
            'Val Acc (%)': round(best_abl_acc, 2)
        })
        print(f"   -> {g_name:42} | Vars: {len(cols):3d} | Val Acc: {best_abl_acc:.2f}%")
        
    df_abl = pd.DataFrame(ablation_results)
    
    # Figure 12 : Ablation study
    plt.figure(figsize=(10, 4.5))
    ax = sns.barplot(data=df_abl, y='Groupe de variables', x='Val Acc (%)', palette='mako')
    plt.title("Étude d'Ablation : Importance Relative des Familles de Capteurs", fontweight='bold')
    plt.xlim(80, 98)
    for i, v in enumerate(df_abl['Val Acc (%)']):
        ax.text(v + 0.2, i, f"{v:.2f}%", va='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig("reports/figures/12_importance_capteurs_ablation.png", dpi=300)
    plt.close()
    print("   -> Figure sauvegardée : reports/figures/12_importance_capteurs_ablation.png")
    
    print("\n" + "=" * 75)
    print("EXPÉRIMENTATIONS DES PARTIES 3 À 7 TERMINÉES AVEC SUCCÈS !")
    print("=" * 75)

if __name__ == "__main__":
    main()
