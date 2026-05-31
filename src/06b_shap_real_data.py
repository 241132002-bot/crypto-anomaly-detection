"""
06b_shap_real_data.py
SHAP анализ върху реалните Ethereum данни
"""

import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import os

os.makedirs('results', exist_ok=True)

print("=" * 55)
print("SHAP АНАЛИЗ — РЕАЛНИ ETHEREUM ДАННИ")
print("=" * 55)

X_test = np.load('data/processed/X_test.npy').astype(np.float32)
y_test = np.load('data/processed/y_test.npy')
clf    = joblib.load('results/random_forest_model.pkl')

feature_names_map = {
    0:'tx_count', 1:'tx_out_count', 2:'tx_in_count', 3:'out_in_ratio',
    4:'total_value_eth', 5:'avg_value_eth', 6:'max_value_eth', 7:'min_value_eth',
    8:'std_value_eth', 9:'median_value_eth', 10:'avg_gas_gwei', 11:'max_gas_gwei',
    12:'std_gas_gwei', 13:'error_rate', 14:'unique_counterparts', 15:'contract_rate',
    16:'avg_interval_sec', 17:'min_interval_sec', 18:'std_interval_sec',
    19:'activity_span_days', 20:'balance_eth', 21:'value_concentration',
}
n_features    = X_test.shape[1]
feature_names = [feature_names_map.get(i, f'f{i+1}') for i in range(n_features)]

print(f"\nХарактеристики: {n_features}")
print(f"Тестови адреси: {len(X_test)}")
print(f"  Аномалии: {y_test.sum()}, Нормални: {(y_test==0).sum()}")

print("\nИзчисляване на SHAP стойности...")
explainer   = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

# Вземи само аномалния клас (индекс 1)
if isinstance(shap_values, list):
    sv = shap_values[1]
elif shap_values.ndim == 3:
    sv = shap_values[:, :, 1]
else:
    sv = shap_values

print(f"SHAP matrix (аномален клас): {sv.shape}")

mean_abs_shap = np.abs(sv).mean(axis=0)
sorted_idx    = np.argsort(mean_abs_shap)[::-1]
top_n         = min(15, n_features)
threshold     = float(mean_abs_shap[sorted_idx[2]])

# ── 1. Bar plot ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
vals    = mean_abs_shap[sorted_idx[:top_n]][::-1]
names   = [feature_names[i] for i in sorted_idx[:top_n]][::-1]
cols    = ['#e74c3c' if float(v) >= threshold else '#3498db' for v in vals]

ax.barh(range(top_n), vals, color=cols, alpha=0.85)
ax.set_yticks(range(top_n))
ax.set_yticklabels(names, fontsize=10)
ax.set_xlabel('Средна абсолютна SHAP стойност', fontsize=11)
ax.set_title('SHAP Feature Importance — Реални Ethereum данни', fontsize=12, fontweight='bold')
red_patch  = mpatches.Patch(color='#e74c3c', alpha=0.85, label='Топ 3 характеристики')
blue_patch = mpatches.Patch(color='#3498db', alpha=0.85, label='Останали')
ax.legend(handles=[red_patch, blue_patch], fontsize=9)
plt.tight_layout()
plt.savefig('results/06_shap_real_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("Запазено: results/06_shap_real_importance.png")

# ── 2. Summary plot ────────────────────────────────────────────────────
plt.figure(figsize=(10, 7))
shap.summary_plot(sv, X_test, feature_names=feature_names,
                  max_display=top_n, show=False, plot_type='dot')
plt.title('SHAP Summary — Реални Ethereum данни', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('results/06_shap_real_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Запазено: results/06_shap_real_summary.png")

# ── 3. Най-аномален адрес ─────────────────────────────────────────────
proba       = clf.predict_proba(X_test)[:, 1]
most_anomal = int(np.argmax(proba))
print(f"\nНай-аномален адрес в тестовия набор:")
print(f"  Risk score: {proba[most_anomal]:.4f}")
print(f"  Реален клас: {'АНОМАЛИЯ' if y_test[most_anomal]==1 else 'НОРМАЛЕН'}")
print(f"\nТоп 5 причини за висок риск:")
addr_shap = sv[most_anomal]
top5_idx  = np.argsort(np.abs(addr_shap))[::-1][:5]
for rank, idx in enumerate(top5_idx, 1):
    direction = "↑ повишава риска" if addr_shap[idx] > 0 else "↓ намалява риска"
    print(f"  {rank}. {feature_names[idx]}: SHAP={addr_shap[idx]:+.4f} ({direction})")

# ── 4. Аномалии vs Нормални ───────────────────────────────────────────
if y_test.sum() > 0 and (y_test==0).sum() > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('SHAP: Аномалии vs Нормални адреси', fontsize=12, fontweight='bold')

    anomal_shap = sv[y_test == 1]
    normal_shap = sv[y_test == 0]
    top5        = sorted_idx[:5]
    x, w        = np.arange(len(top5)), 0.35

    axes[0].bar(x-w/2, np.abs(anomal_shap).mean(axis=0)[top5], w,
                label='Аномалии', color='#e74c3c', alpha=0.8)
    axes[0].bar(x+w/2, np.abs(normal_shap).mean(axis=0)[top5], w,
                label='Нормални', color='#2ecc71', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([feature_names[i] for i in top5], rotation=15, ha='right')
    axes[0].set_title('Средна |SHAP| по клас (Топ 5)')
    axes[0].set_ylabel('Средна абсолютна SHAP стойност')
    axes[0].legend()

    axes[1].hist(proba[y_test==0], bins=10, alpha=0.7, color='#2ecc71', label='Нормални')
    axes[1].hist(proba[y_test==1], bins=5,  alpha=0.7, color='#e74c3c', label='Аномалии')
    axes[1].axvline(0.5, color='black', linestyle='--', label='Праг 0.5')
    axes[1].set_xlabel('Risk Score')
    axes[1].set_ylabel('Брой адреси')
    axes[1].set_title('Разпределение на Risk Score')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('results/06_shap_real_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Запазено: results/06_shap_real_comparison.png")

print("\n" + "="*55)
print("SHAP АНАЛИЗ ЗАВЪРШЕН!")
print("="*55)
print("Стартирай: python src/07_early_warning.py")
