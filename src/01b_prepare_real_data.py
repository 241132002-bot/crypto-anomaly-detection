"""
01b_prepare_real_data.py
Подготовка на реални данни за обучение на моделите
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

os.makedirs('data/processed', exist_ok=True)

print("=" * 55)
print("ПОДГОТОВКА НА РЕАЛНИ ДАННИ")
print("=" * 55)

df = pd.read_csv('data/raw/real_transactions.csv')
print(f"\nЗаредени: {len(df)} адреса")
print(f"  Аномалии: {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
print(f"  Нормални: {(df['label']==0).sum()}")

# Характеристики
feature_cols = [c for c in df.columns if c not in ['address', 'label']]
print(f"\nХарактеристики: {len(feature_cols)}")
print(f"  {feature_cols}")

# Попълване на липсващи стойности
df[feature_cols] = df[feature_cols].fillna(0)

X = df[feature_cols].values.astype(np.float32)
y = df['label'].values.astype(np.int64)

# Нормализация
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделяне — при малко данни 70/15/15
if len(df) >= 30:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
else:
    # При много малко данни — само train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42
    )
    X_val, y_val = X_test, y_test
    print("  Внимание: малко данни — val=test набор")

print(f"\nРазделяне:")
print(f"  Обучение:  {len(X_train)}")
print(f"  Валидация: {len(X_val)}")
print(f"  Тест:      {len(X_test)}")

# ЗАПАЗВАНЕ С _real
np.save('data/processed/X_train_real.npy', X_train)
np.save('data/processed/X_val_real.npy',   X_val)
np.save('data/processed/X_test_real.npy',  X_test)
np.save('data/processed/y_train_real.npy', y_train)
np.save('data/processed/y_val_real.npy',   y_val)
np.save('data/processed/y_test_real.npy',  y_test)
joblib.dump(scaler, 'data/processed/scaler_real.pkl')
joblib.dump(feature_cols, 'data/processed/feature_cols_real.pkl')

print(f"\n✅ Данните са запазени в data/processed/")
print(f"   X_train_real.npy, X_val_real.npy, X_test_real.npy")
print(f"   y_train_real.npy, y_val_real.npy, y_test_real.npy")
print(f"   scaler_real.pkl, feature_cols_real.pkl")

print("\n📊 Стартирай: python src/02_isolation_forest.py")