import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_har_data(data_dir="data/HAR"):
    """
    Charge les fichiers train.csv et test.csv du dataset HAR.
    """
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df

def prepare_features_and_labels(df, label_encoder=None):
    """
    Sépare les variables explicatives (561 capteurs) et la cible (Activity).
    Remarque importante pour la soutenance :
    - La colonne 'subject' est retirée car c'est un identifiant d'individu.
      L'inclure fausserait la généralisation à de nouveaux individus.
    - 'Activity' est encodée de 0 à 5 pour PyTorch CrossEntropyLoss.
    """
    # On isole les 561 features de capteurs
    feature_cols = [col for col in df.columns if col not in ['subject', 'Activity']]
    X = df[feature_cols].values.astype(np.float32)
    
    # Encodage de l'activité (chaîne de caractères -> entier)
    if label_encoder is None:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df['Activity'])
    else:
        y = label_encoder.transform(df['Activity'])
        
    y = y.astype(np.int64)
    return X, y, label_encoder, feature_cols

def split_by_subject(train_df, val_subjects=None):
    """
    Découpage par volontaires entiers (Group Split).
    Par défaut, on réserve 4 volontaires pour la validation (ex: 1, 3, 5, 7).
    Justification : évite la fuite de données liée au chevauchement (50%)
    des fenêtres temporelles consécutives d'un même individu.
    """
    if val_subjects is None:
        # 4 sujets représentatifs parmi les 21 sujets du train
        val_subjects = [1, 3, 5, 7]
        
    train_split = train_df[~train_df['subject'].isin(val_subjects)].copy()
    val_split = train_df[train_df['subject'].isin(val_subjects)].copy()
    
    return train_split, val_split

def split_random(train_df, val_ratio=0.2, random_state=42):
    """
    Découpage aléatoire ligne par ligne (utilisé pour la comparaison en partie 1.3).
    """
    train_split, val_split = train_test_split(
        train_df,
        test_size=val_ratio,
        random_state=random_state,
        stratify=train_df['Activity']
    )
    return train_split, val_split

def create_dataloaders(X_train, y_train, X_val, y_val, batch_size=64):
    """
    Convertit les arrays numpy en Tensors PyTorch et crée les DataLoaders standards.
    """
    train_tensor = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_tensor = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

