"""
06_shap_analysis.py
SHAP анализ — интерпретация на Random Forest
"""

import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

os.makedirs('results', exist_ok=True)

print("=" * 50)
print("СТЪПКА 6: SHAP анализ")
print("=" * 50)

X_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')
clf    = joblib.load('results/random_forest_model.pkl')

feature_names = [f'f{i}' for i in range(1, 165)]

print("\nИзчисляване на SHAP стойности (~1-2 мин.)...")
explainer   = shap.TreeExplainer(clf)
X_sample    = X_test[:500]
shap_values = explainer.shap_values(X_sample)
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

# Summary plot
plt.figure(figsize=(10, 8))
shap.summary_plot(sv, X_sample, feature_names=feature_names,
                  max_display=20, show=False)
plt.title('SHAP Summary — Топ 20 характеристики')
plt.tight_layout()
plt.savefig('results/05_shap_summary.png', dpi=150, bbox_inches='tight')
plt.show()

# Bar plot
plt.figure(figsize=(10, 6))
shap.summary_plot(sv, X_sample, feature_names=feature_names,
                  plot_type='bar', max_display=15, show=False)
plt.title('SHAP Feature Importance')
plt.tight_layout()
plt.savefig('results/05_shap_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nSHAP графиките са запазени в results/")
print("\n" + "=" * 50)
print("ПРОЕКТЪТ Е ЗАВЪРШЕН!")
print("=" * 50)
print("\nВсички резултати са в папка results/:")
print("  01_isolation_forest.png")
print("  02_random_forest.png")
print("  03_graphsage.png")
print("  04_ensemble_comparison.png  ← главна графика")
print("  05_shap_summary.png")
print("  05_shap_importance.png")
print("  final_results.json          ← всички метрики")
