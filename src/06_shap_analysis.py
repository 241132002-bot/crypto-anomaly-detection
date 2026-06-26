import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

print("SHAP анализ...")

X_test = np.load("data/processed/X_test.npy")
clf    = joblib.load("results/random_forest_model.pkl")

feature_names = [f"f{i}" for i in range(1, 165)]

print("Изчисляване на SHAP стойности...")
explainer = shap.TreeExplainer(clf)
X_sample  = X_test[:200]

# Използваме predict_proba директно
shap_values = explainer(X_sample)

# Вземаме стойностите за клас 1 (аномалия)
if hasattr(shap_values, 'values'):
    sv = shap_values.values
    if sv.ndim == 3:
        sv = sv[:, :, 1]
else:
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    if sv.ndim == 3:
        sv = sv[:, :, 1]

print(f"SHAP shape: {sv.shape}")

# Summary plot
plt.figure(figsize=(14, 10))
shap.summary_plot(sv, X_sample,
                  feature_names=feature_names,
                  max_display=20,
                  show=False)
plt.title("SHAP Summary — Топ 20 характеристики", fontsize=14)
plt.tight_layout()
plt.savefig("results/05_shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Summary plot запазен!")

# Bar plot
plt.figure(figsize=(14, 8))
shap.summary_plot(sv, X_sample,
                  feature_names=feature_names,
                  plot_type="bar",
                  max_display=15,
                  show=False)
plt.title("SHAP Feature Importance — Топ 15", fontsize=14)
plt.tight_layout()
plt.savefig("results/05_shap_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Importance plot запазен!")

print("Готово! Графиките са в results/")