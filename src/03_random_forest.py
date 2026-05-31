"""
03_random_forest.py
Random Forest — супервизирана класификация
"""

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, precision_recall_curve,
                              f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('results', exist_ok=True)

print("=" * 50)
print("СТЪПКА 3: Random Forest")
print("=" * 50)

X_train = np.load('data/processed/X_train.npy')
X_val   = np.load('data/processed/X_val.npy')
X_test  = np.load('data/processed/X_test.npy')
y_train = np.load('data/processed/y_train.npy')
y_val   = np.load('data/processed/y_val.npy')
y_test  = np.load('data/processed/y_test.npy')

print("\nОбучение (може да отнеме ~1 мин.)...")
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# Валидация
y_val_pred = clf.predict(X_val)
val_f1 = f1_score(y_val, y_val_pred)
print(f"Валидационен F1-Score: {val_f1:.4f}")

# Тест
y_pred = clf.predict(X_test)
scores = clf.predict_proba(X_test)[:, 1]

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

# Feature Importance
importances = clf.feature_importances_
top_idx = np.argsort(importances)[::-1][:15]
print("\nТоп 15 характеристики:")
for rank, idx in enumerate(top_idx, 1):
    print(f"  {rank:2d}. f{idx+1:3d}: {importances[idx]:.4f}")

# Графики
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Random Forest', fontsize=14, fontweight='bold')

sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[0],
            xticklabels=['Нормална', 'Аномалия'],
            yticklabels=['Нормална', 'Аномалия'])
axes[0].set_title('Confusion Matrix')
axes[0].set_ylabel('Реален')
axes[0].set_xlabel('Предвиден')

fpr, tpr, _ = roc_curve(y_test, scores)
axes[1].plot(fpr, tpr, 'g-', lw=2, label=f'AUC = {auc:.3f}')
axes[1].plot([0,1],[0,1],'k--', lw=1)
axes[1].set_title('ROC крива')
axes[1].set_xlabel('FPR')
axes[1].set_ylabel('TPR')
axes[1].legend()

feat_names = [f'f{i+1}' for i in top_idx]
axes[2].barh(range(15), importances[top_idx[::-1]], color='green', alpha=0.7)
axes[2].set_yticks(range(15))
axes[2].set_yticklabels(feat_names[::-1], fontsize=8)
axes[2].set_title('Feature Importance (Топ 15)')
axes[2].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('results/02_random_forest.png', dpi=150)
plt.show()

joblib.dump(clf, 'results/random_forest_model.pkl')
np.save('results/rf_scores.npy', scores)
np.save('results/rf_predictions.npy', y_pred)

print("\nЗапазено. Стартирай 04_graphsage.py")
