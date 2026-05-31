"""
04_graphsage.py
GraphSAGE — адаптиран за реални данни
Изгражда граф на база сходство между адресите
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, f1_score)
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

os.makedirs('results', exist_ok=True)

print("=" * 50)
print("СТЪПКА 4: GraphSAGE")
print("=" * 50)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Устройство: {device}")

# Зареждане
X_train = np.load('data/processed/X_train.npy').astype(np.float32)
X_val   = np.load('data/processed/X_val.npy').astype(np.float32)
X_test  = np.load('data/processed/X_test.npy').astype(np.float32)
y_train = np.load('data/processed/y_train.npy').astype(np.int64)
y_val   = np.load('data/processed/y_val.npy').astype(np.int64)
y_test  = np.load('data/processed/y_test.npy').astype(np.int64)

all_X = np.vstack([X_train, X_val, X_test])
all_y = np.concatenate([y_train, y_val, y_test])
n_train, n_val = len(X_train), len(X_val)
n_total = len(all_X)

print(f"\nОбщо възли: {n_total}")

# ── Изграждане на граф чрез k-NN сходство ─────────────────────────────
# Свързваме всеки адрес с k най-близки по характеристики
print("\nИзграждане на граф чрез k-NN сходство (k=5)...")
k = 5
nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine', algorithm='brute')
nbrs.fit(all_X)
distances, indices = nbrs.kneighbors(all_X)

src_nodes, dst_nodes = [], []
for i in range(n_total):
    for j in indices[i][1:]:  # пропускаме самия възел (индекс 0)
        src_nodes.append(i)
        dst_nodes.append(int(j))

# Добавяме и обратни ръбове (undirected)
edge_index = torch.tensor(
    [src_nodes + dst_nodes, dst_nodes + src_nodes],
    dtype=torch.long
)

print(f"  Ръбове: {edge_index.shape[1]:,} (k={k} за {n_total} възла)")

# Проверка
max_idx = edge_index.max().item()
print(f"  Макс индекс в edge_index: {max_idx} (трябва < {n_total})")
assert max_idx < n_total, f"Грешка: индекс {max_idx} >= {n_total}"

# Masks
train_mask = torch.zeros(n_total, dtype=torch.bool)
val_mask   = torch.zeros(n_total, dtype=torch.bool)
test_mask  = torch.zeros(n_total, dtype=torch.bool)
train_mask[:n_train] = True
val_mask[n_train:n_train+n_val] = True
test_mask[n_train+n_val:] = True

data = Data(
    x          = torch.tensor(all_X, dtype=torch.float32),
    edge_index = edge_index,
    y          = torch.tensor(all_y, dtype=torch.long),
    train_mask = train_mask,
    val_mask   = val_mask,
    test_mask  = test_mask
).to(device)

print(f"  Граф изграден: {data.num_nodes} възла, {data.num_edges} ръба")

# ── Модел ─────────────────────────────────────────────────────────────
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch, dropout=0.3):
        super().__init__()
        self.conv1   = SAGEConv(in_ch, hid_ch)
        self.conv2   = SAGEConv(hid_ch, hid_ch)
        self.conv3   = SAGEConv(hid_ch, out_ch)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv3(x, edge_index)

model     = GraphSAGE(data.num_features, 64, 2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

pos_weight = float((1 - y_train.mean()) / y_train.mean())
pw         = torch.tensor([1.0, pos_weight], dtype=torch.float32).to(device)
criterion  = torch.nn.CrossEntropyLoss(weight=pw)

# ── Обучение ───────────────────────────────────────────────────────────
print("\nОбучение (50 епохи)...")
best_val_f1, best_epoch = 0, 0
train_losses, val_f1s   = [], []

for epoch in range(1, 51):
    model.train()
    optimizer.zero_grad()
    out  = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        vp  = out[data.val_mask].argmax(dim=1).cpu().numpy()
        vt  = data.y[data.val_mask].cpu().numpy()
        vf1 = f1_score(vt, vp, zero_division=0)
        val_f1s.append(vf1)
        if vf1 > best_val_f1:
            best_val_f1, best_epoch = vf1, epoch
            torch.save(model.state_dict(), 'results/graphsage_best.pt')

    if epoch % 10 == 0:
        print(f"  Епоха {epoch:3d} | Loss: {loss.item():.4f} | Val F1: {vf1:.4f}")

print(f"\nНай-добра епоха: {best_epoch}  (Val F1: {best_val_f1:.4f})")

# ── Оценка ─────────────────────────────────────────────────────────────
model.load_state_dict(torch.load('results/graphsage_best.pt'))
model.eval()
with torch.no_grad():
    out    = model(data.x, data.edge_index)
    probs  = F.softmax(out[data.test_mask], dim=1).cpu().numpy()
    y_pred = probs.argmax(axis=1)
    scores = probs[:, 1]

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

# ── Графики ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('GraphSAGE — Реални Ethereum данни (k-NN граф)', fontsize=13, fontweight='bold')

sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes[0],
            xticklabels=['Нормална', 'Аномалия'],
            yticklabels=['Нормална', 'Аномалия'])
axes[0].set_title('Confusion Matrix')

fpr, tpr, _ = roc_curve(y_test, scores)
axes[1].plot(fpr, tpr, color='orange', lw=2, label=f'AUC = {auc:.3f}')
axes[1].plot([0,1],[0,1],'k--')
axes[1].set_title('ROC крива')
axes[1].legend()

axes[2].plot(train_losses, label='Train Loss', color='steelblue')
axes[2].plot(val_f1s, label='Val F1', color='orange')
axes[2].axvline(best_epoch-1, color='r', linestyle='--', label=f'Best ep.{best_epoch}')
axes[2].set_title('Крива на обучение')
axes[2].legend()

plt.tight_layout()
plt.savefig('results/03_graphsage.png', dpi=150)
plt.show()

np.save('results/gs_scores.npy', scores)
np.save('results/gs_predictions.npy', y_pred)
print("\nЗапазено. Стартирай 05_ensemble.py")
