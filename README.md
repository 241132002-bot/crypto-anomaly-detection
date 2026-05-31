# Дипломен проект — Откриване на аномалии в криптоекосистеми

## Структура на проекта
```
diploma_project/
├── .vscode/
│   ├── settings.json       ← VS Code настройки
│   ├── launch.json         ← Run конфигурации (F5)
│   └── extensions.json     ← Препоръчани разширения
├── data/
│   ├── raw/                ← ПОСТАВИ ТУК трите CSV файла
│   └── processed/          ← автоматично се създава
├── src/
│   ├── 01_data_preparation.py
│   ├── 02_isolation_forest.py
│   ├── 03_random_forest.py
│   ├── 04_graphsage.py
│   ├── 05_ensemble.py
│   └── 06_shap_analysis.py
├── results/                ← автоматично се създава
├── setup.bat               ← еднократна настройка
└── requirements.txt
```

---

## Първоначална настройка (само веднъж)

### 1. Свали Elliptic Dataset
https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

Постави трите файла в `data/raw/`:
- `elliptic_txs_features.csv`
- `elliptic_txs_classes.csv`
- `elliptic_txs_edgelist.csv`

### 2. Стартирай setup.bat
Двойно кликни върху `setup.bat` — ще създаде виртуална среда и ще инсталира всичко.

### 3. Избери Python interpreter в VS Code
`Ctrl+Shift+P` → "Python: Select Interpreter" → избери `.venv`

---

## Стартиране

**Вариант А — от терминала (препоръчително):**
```bash
cd src
python 01_data_preparation.py
python 02_isolation_forest.py
python 03_random_forest.py
python 04_graphsage.py
python 05_ensemble.py
python 06_shap_analysis.py
```

**Вариант Б — от VS Code с F5:**
Отвори файла → натисни F5 → избери конфигурацията от launch.json

---

## Очаквано време за изпълнение
| Скрипт | Време |
|--------|-------|
| 01 | ~30 сек |
| 02 | ~1 мин |
| 03 | ~2 мин |
| 04 | ~10 мин (CPU) |
| 05 | ~30 сек |
| 06 | ~2 мин |

---

## Резултати
След изпълнение в `results/` ще намериш:
| Файл | Описание |
|------|----------|
| `01_isolation_forest.png` | Confusion Matrix, ROC, PR криви |
| `02_random_forest.png` | + Feature Importance |
| `03_graphsage.png` | + Learning curve |
| `04_ensemble_comparison.png` | **Главна графика за дипломата** |
| `05_shap_summary.png` | SHAP интерпретация |
| `final_results.json` | Всички метрики в числа |
