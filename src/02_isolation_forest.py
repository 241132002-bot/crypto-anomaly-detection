"""
02_isolation_forest.py
Isolation Forest — несупервизирано откриване на аномалии
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

print("=" * 50)
print("СТЪПКА 2: Isolation Forest")
print("=" * 50)

X_train = np.load('data/processed/X_train.npy')
X_test  = np.load('data/processed/X_test.npy')
y_train = np.load('data/processed/y_train.npy')
y_test  = np.load('data/processed/y_test.npy')

contamination = float(y_train.mean())
print(f"\nОчакван дял аномалии: {contamination:.3f}")
print("Обучение...")

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
scores = (scores - scores.min()) / (scores.max() - scores.min())

# Праг базиран на очаквания дял аномалии
threshold = np.percentile(scores, (1 - contamination) * 100)
y_pred = (scores >= threshold).astype(int)

print(f"Праг: {threshold:.4f}")
print(f"Открити аномалии: {y_pred.sum()}")

print("\n" + "=" * 50)
print("РЕЗУЛТАТИ")
print("=" * 50)
print(classification_report(y_test, y_pred,
                             target_names=['Нормална', 'Аномалия']))

auc = roc_auc_score(y_test, scores)
print(f"ROC-AUC: {auc:.4f}")

cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

# Графики
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Isolation Forest', fontsize=14, fontweight='bold')

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Нормална', 'Аномалия'],
            yticklabels=['Нормална', 'Аномалия'])
axes[0].set_title('Confusion Matrix')
axes[0].set_ylabel('Реален')
axes[0].set_xlabel('Предвиден')

fpr, tpr, _ = roc_curve(y_test, scores)
axes[1].plot(fpr, tpr, 'b-', lw=2, label=f'AUC = {auc:.3f}')
axes[1].plot([0,1],[0,1],'k--', lw=1)
axes[1].set_title('ROC крива')
axes[1].set_xlabel('FPR')
axes[1].set_ylabel('TPR')
axes[1].legend()

prec, rec, _ = precision_recall_curve(y_test, scores)
axes[2].plot(rec, prec, 'b-', lw=2)
axes[2].set_title('Precision-Recall')
axes[2].set_xlabel('Recall')
axes[2].set_ylabel('Precision')

plt.tight_layout()
plt.savefig('results/01_isolation_forest.png', dpi=150)
plt.show()

joblib.dump(clf, 'results/isolation_forest_model.pkl')
np.save('results/if_scores.npy', scores)
np.save('results/if_predictions.npy', y_pred)

print("\nЗапазено. Стартирай 03_random_forest.py")
