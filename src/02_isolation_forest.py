"""
02_isolation_forest.py
Isolation Forest — несупервизирано откриване на аномалии върху реални Ethereum данни
"""

import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('results', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

print("=" * 60)
print("СТЪПКА 2: Isolation Forest — Реални Ethereum данни")
print("=" * 60)

# ============================================================
# ЗАРЕЖДАНЕ НА ДАННИТЕ
# ============================================================
try:
    X_train = np.load('data/processed/X_train_real.npy')
    X_test = np.load('data/processed/X_test_real.npy')
    y_train = np.load('data/processed/y_train_real.npy')
    y_test = np.load('data/processed/y_test_real.npy')
    print(f"\n✅ Заредени данни:")
    print(f"   Обучение: {X_train.shape}")
    print(f"   Тест: {X_test.shape}")
    print(f"   Аномалии в обучение: {y_train.sum()} ({(y_train.sum()/len(y_train))*100:.1f}%)")
    print(f"   Аномалии в тест: {y_test.sum()} ({(y_test.sum()/len(y_test))*100:.1f}%)")
except FileNotFoundError as e:
    print(f"❌ Грешка: {e}")
    print("   Моля, първо стартирайте: python src/01b_prepare_real_data.py")
    exit(1)

# ============================================================
# ОБУЧЕНИЕ
# ============================================================
contamination = float(y_train.mean())
print(f"\nОчакван дял аномалии: {contamination:.3f}")
print("Обучение на Isolation Forest...")

clf = IsolationForest(
    n_estimators=200,
    contamination=contamination,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train)

# score_samples: по-ниски стойности = по-аномални
raw_scores = clf.score_samples(X_test)

# Нормализираме и обръщаме: по-висок = по-аномален
scores = -raw_scores
scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

# Праг базиран на очаквания дял аномалии
threshold = np.percentile(scores, (1 - contamination) * 100)
y_pred = (scores >= threshold).astype(int)

print(f"\nПраг: {threshold:.4f}")
print(f"Открити аномалии в тест: {y_pred.sum()}")

# ============================================================
# РЕЗУЛТАТИ
# ============================================================
print("\n" + "=" * 60)
print("РЕЗУЛТАТИ")
print("=" * 60)
print(classification_report(y_test, y_pred,
                             target_names=['Нормална', 'Аномалия']))

auc = roc_auc_score(y_test, scores)
print(f"\nROC-AUC: {auc:.4f}")

cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

# ============================================================
# ГРАФИКИ
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Isolation Forest — Реални Ethereum данни', fontsize=14, fontweight='bold')

# 1. Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Нормална', 'Аномалия'],
            yticklabels=['Нормална', 'Аномалия'])
axes[0].set_title('Confusion Matrix')
axes[0].set_ylabel('Реален')
axes[0].set_xlabel('Предвиден')

# 2. ROC крива
fpr, tpr, _ = roc_curve(y_test, scores)
axes[1].plot(fpr, tpr, 'b-', lw=2, label=f'AUC = {auc:.3f}')
axes[1].plot([0,1],[0,1],'k--', lw=1)
axes[1].set_title('ROC крива')
axes[1].set_xlabel('FPR')
axes[1].set_ylabel('TPR')
axes[1].legend()

# 3. Precision-Recall крива
prec, rec, _ = precision_recall_curve(y_test, scores)
axes[2].plot(rec, prec, 'b-', lw=2)
axes[2].set_title('Precision-Recall')
axes[2].set_xlabel('Recall')
axes[2].set_ylabel('Precision')

plt.tight_layout()
plt.savefig('results/figures/isolation_forest_real.png', dpi=150)
print(f"\n✅ Графиката е запазена: results/figures/isolation_forest_real.png")

# ============================================================
# ЗАПАЗВАНЕ
# ============================================================
joblib.dump(clf, 'data/processed/isolation_forest.pkl')
np.save('data/processed/if_scores_real.npy', scores)
np.save('data/processed/if_predictions_real.npy', y_pred)

print("\n✅ Моделът е запазен в: data/processed/isolation_forest.pkl")
print("📊 Стартирайте: python src/03_random_forest.py")