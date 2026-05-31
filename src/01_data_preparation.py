"""
01_data_preparation.py
Зареждане и подготовка на Elliptic Dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

DATA_RAW = 'data/raw'
DATA_OUT = 'data/processed'
os.makedirs(DATA_OUT, exist_ok=True)

print("=" * 50)
print("СТЪПКА 1: Подготовка на данните")
print("=" * 50)

# Зареждане
print("\nЗареждане на CSV файлове...")
features = pd.read_csv(f'{DATA_RAW}/elliptic_txs_features.csv', header=None)
classes  = pd.read_csv(f'{DATA_RAW}/elliptic_txs_classes.csv')
edges    = pd.read_csv(f'{DATA_RAW}/elliptic_txs_edgelist.csv')

n_cols = len(features.columns)
features.columns = ['txId', 'time_step'] + [f'f{i}' for i in range(1, n_cols - 1)]

print(f"  Транзакции общо:  {len(features):,}")
print(f"  Ръбове в графа:   {len(edges):,}")
print(f"  Разпределение:\n{classes['class'].value_counts().to_string()}")

# Обединяване и филтриране
df      = features.merge(classes, on='txId')
labeled = df[df['class'] != 'unknown'].copy()
labeled['label'] = labeled['class'].isin(['1', 'suspicious']).astype(int)
print(f"\nЕтикетирани транзакции: {len(labeled):,}")
print(f"  Аномалии:  {labeled['label'].sum():,} ({labeled['label'].mean()*100:.1f}%)")
print(f"  Нормални:  {(labeled['label']==0).sum():,}")

# Характеристики
feature_cols = [f'f{i}' for i in range(1, 165)]
X = labeled[feature_cols].values
y = labeled['label'].values

# Нормализация
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделяне 70 / 15 / 15
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"\nРазделяне на набора:")
print(f"  Обучение:  {len(X_train):,}")
print(f"  Валидация: {len(X_val):,}")
print(f"  Тест:      {len(X_test):,}")

# Запазване
np.save(f'{DATA_OUT}/X_train.npy', X_train)
np.save(f'{DATA_OUT}/X_val.npy',   X_val)
np.save(f'{DATA_OUT}/X_test.npy',  X_test)
np.save(f'{DATA_OUT}/y_train.npy', y_train)
np.save(f'{DATA_OUT}/y_val.npy',   y_val)
np.save(f'{DATA_OUT}/y_test.npy',  y_test)

joblib.dump(scaler, f'{DATA_OUT}/scaler.pkl')
joblib.dump(
    labeled[['txId', 'label']].reset_index(drop=True),
    f'{DATA_OUT}/labeled_meta.pkl'
)

labeled_ids     = set(labeled['txId'].values)
edges.columns = ['txId1', 'txId2']
edges_filtered  = edges[
    edges['txId1'].isin(labeled_ids) & edges['txId2'].isin(labeled_ids)
]
edges_filtered.to_csv(f'{DATA_OUT}/edges_labeled.csv', index=False)

print(f"\nЗапазени файлове в {DATA_OUT}/")
print("Готово! Стартирай 02_isolation_forest.py")
