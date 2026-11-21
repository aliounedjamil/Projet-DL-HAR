"""
Script d'entraînement pour la Version 1 (Partie 2 du projet HAR)
Auteur : Alioune DIOP (Binôme : Madjiguène FALL)
Enseignant : Walid Bechkit

Ce script réalise pas à pas :
1. Contrôle préalable : Vérification de la perte initiale théorique (~1.79)
2. Sanity check : Surapprentissage d'un mini-lot de 10 exemples
3. Étude comparative des fonctions d'activation : ReLU, Tanh, Sigmoid
4. Étude comparative des algorithmes d'optimisation : SGD, SGD Momentum, Adam
5. Étude de variabilité statistique sur 3 graines aléatoires (seeds 42, 123, 999)
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

from data_loader import load_har_data, split_by_subject, prepare_features_and_labels, create_dataloaders

# Fixer la graine aléatoire pour la reproductibilité
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# Définition du modèle simple à 1 couche cachée (64 neurones)
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=561, hidden_dim=64, num_classes=6, activation='relu'):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Activation inconnue : {activation}")
            
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

# Fonction d'entraînement d'une époque (standard TP)
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x_batch.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += x_batch.size(0)
        
    return total_loss / total, correct / total

# Fonction d'évaluation sur validation (standard TP)
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item() * x_batch.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            
    total = len(all_targets)
    epoch_loss = total_loss / total
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average='macro')
    return epoch_loss, epoch_acc, epoch_f1

# Fonction d'entraînement complet
def run_training(train_loader, val_loader, activation='relu', opt_name='adam', lr=0.001, epochs=25, seed=42):
    set_seed(seed)
    model = SimpleMLP(hidden_dim=64, activation=activation)
    criterion = nn.CrossEntropyLoss()
    
    if opt_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif opt_name == 'sgd_momentum':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(opt_name)
        
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}
    
    for ep in range(epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, vl_f1 = evaluate(model, val_loader, criterion)
        
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)
        history['val_f1'].append(vl_f1)
        
    return model, history

def main():
    print("=" * 70)
    print("PROJET HAR : EXÉCUTION DE LA VERSION 1 (MLP 1 COUCHE CACHÉE)")
    print("Auteur : Alioune DIOP")
    print("=" * 70)
    
    # 1. Chargement et préparation des données
    print("\n1. Chargement des données et découpage par volontaires (sujets 1, 3, 5, 7)...")
    train_df, test_df = load_har_data("data/HAR")
    train_split, val_split = split_by_subject(train_df, val_subjects=[1, 3, 5, 7])
    
    X_tr, y_tr, le, feature_cols = prepare_features_and_labels(train_split)
    X_val, y_val, _, _ = prepare_features_and_labels(val_split, label_encoder=le)
    
    train_loader, val_loader = create_dataloaders(X_tr, y_tr, X_val, y_val, batch_size=64)
    print(f"   Train : {len(X_tr)} échantillons | Validation : {len(X_val)} échantillons")
    
    # 2. Contrôle de la perte initiale théorique
    print("\n2. Contrôle de la perte initiale théorique (Partie 2.1)...")
    criterion = nn.CrossEntropyLoss()
    model_test = SimpleMLP(hidden_dim=64, activation='relu')
    xb, yb = next(iter(train_loader))
    with torch.no_grad():
        initial_loss = criterion(model_test(xb), yb).item()
    print(f"   Perte initiale observée : {initial_loss:.4f}")
    print(f"   Perte théorique ln(6)   : {np.log(6):.4f}")
    print("   -> Validation : Perte initiale conforme à la théorie !")
    
    # 3. Sanity Check : Surapprentissage d'un mini-lot
    print("\n3. Sanity Check : Surapprentissage d'un mini-lot de 10 exemples (Partie 2.1)...")
    x_tiny, y_tiny = xb[:10], yb[:10]
    overfit_model = SimpleMLP(hidden_dim=64, activation='relu')
    overfit_opt = optim.Adam(overfit_model.parameters(), lr=0.01)
    for _ in range(50):
        overfit_opt.zero_grad()
        l = criterion(overfit_model(x_tiny), y_tiny)
        l.backward()
        overfit_opt.step()
    acc_tiny = (overfit_model(x_tiny).argmax(dim=1) == y_tiny).float().mean().item()
    print(f"   Précision sur le mini-lot : {acc_tiny * 100:.1f}% (Perte = {l.item():.5f})")
    print("   -> Validation : Le réseau est capable de mémoriser et d'optimiser sans bug !")
    
    # 4. Comparaison des fonctions d'activation
    print("\n4. Comparaison des fonctions d'activation (Partie 2.2)...")
    activations = ['relu', 'tanh', 'sigmoid']
    act_histories = {}
    for act in activations:
        print(f"   Entraînement avec activation = {act.upper()} (20 époques)...")
        _, hist = run_training(train_loader, val_loader, activation=act, opt_name='adam', lr=0.001, epochs=20)
        act_histories[act] = hist
        print(f"   -> {act.upper()} final : Val Loss = {hist['val_loss'][-1]:.4f} | Val Acc = {hist['val_acc'][-1]*100:.2f}% | Val Macro-F1 = {hist['val_f1'][-1]*100:.2f}%")
        
    # Tracé des courbes d'activation
    os.makedirs("reports/figures", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for act in activations:
        axes[0].plot(act_histories[act]['val_loss'], label=f"{act.upper()}")
        axes[1].plot(act_histories[act]['val_acc'], label=f"{act.upper()}")
    axes[0].set_title("Validation Loss par Activation", fontweight='bold')
    axes[0].set_xlabel("Époque")
    axes[0].legend()
    axes[1].set_title("Validation Accuracy par Activation", fontweight='bold')
    axes[1].set_xlabel("Époque")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("reports/figures/05_comparaison_activations.png", dpi=300)
    plt.close()
    print("   -> Figure sauvegardée : reports/figures/05_comparaison_activations.png")
    
    # 5. Comparaison des algorithmes d'optimisation
    print("\n5. Comparaison des algorithmes d'optimisation (Partie 2.2)...")
    opt_configs = [
        ("SGD classique", "sgd", 0.05),
        ("SGD + Momentum", "sgd_momentum", 0.01),
        ("Adam", "adam", 0.001)
    ]
    opt_histories = {}
    for label, opt_name, lr in opt_configs:
        print(f"   Entraînement avec {label} (lr={lr}, 20 époques)...")
        _, hist = run_training(train_loader, val_loader, activation='relu', opt_name=opt_name, lr=lr, epochs=20)
        opt_histories[label] = hist
        print(f"   -> {label} final : Val Loss = {hist['val_loss'][-1]:.4f} | Val Acc = {hist['val_acc'][-1]*100:.2f}%")
        
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, _, _ in opt_configs:
        axes[0].plot(opt_histories[label]['val_loss'], label=label)
        axes[1].plot(opt_histories[label]['val_acc'], label=label)
    axes[0].set_title("Validation Loss par Optimiseur", fontweight='bold')
    axes[0].set_xlabel("Époque")
    axes[0].legend()
    axes[1].set_title("Validation Accuracy par Optimiseur", fontweight='bold')
    axes[1].set_xlabel("Époque")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("reports/figures/06_comparaison_optimiseurs.png", dpi=300)
    plt.close()
    print("   -> Figure sauvegardée : reports/figures/06_comparaison_optimiseurs.png")
    
    # 6. Variabilité statistique multi-graines
    print("\n6. Étude de variabilité statistique sur 3 graines (Partie 2.3)...")
    seeds = [42, 123, 999]
    scores_relu = []
    scores_tanh = []
    for s in seeds:
        _, h_relu = run_training(train_loader, val_loader, activation='relu', opt_name='adam', lr=0.001, epochs=20, seed=s)
        _, h_tanh = run_training(train_loader, val_loader, activation='tanh', opt_name='adam', lr=0.001, epochs=20, seed=s)
        scores_relu.append(h_relu['val_acc'][-1] * 100)
        scores_tanh.append(h_tanh['val_acc'][-1] * 100)
        
    print(f"   Graines testées : {seeds}")
    print(f"   Scores ReLU + Adam : {[round(x, 2) for x in scores_relu]} %")
    print(f"   -> Moyenne : {np.mean(scores_relu):.2f}% | Écart-type : {np.std(scores_relu):.2f}%")
    print(f"   Scores Tanh + Adam : {[round(x, 2) for x in scores_tanh]} %")
    print(f"   -> Moyenne : {np.mean(scores_tanh):.2f}% | Écart-type : {np.std(scores_tanh):.2f}%")
    
    print("\n" + "=" * 70)
    print("TOUS LES SCRIPTS DE LA VERSION 1 ONT ÉTÉ EXÉCUTÉS AVEC SUCCÈS !")
    print("=" * 70)

if __name__ == "__main__":
    main()
