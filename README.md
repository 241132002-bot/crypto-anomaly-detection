# Early Warning System — UI файлове
## Crypto Anomaly Detection | Магистърска теза 2026

Три варианта за стартиране на dashboard-а:

---

## 1. dashboard.html — без инсталация
Отвори директно в браузъра (двойно кликване).

Функции:
- Зареди твоя `monitoring_log.csv` чрез бутона
- Demo данни при първо отваряне
- Търсене, филтър, сортиране по риск score
- SHAP анализ при клик върху адрес
- Експорт на CSV

---

## 2. streamlit_app.py — Python UI
```bash
pip install streamlit plotly pandas
streamlit run streamlit_app.py
```
Отваря се на: http://localhost:8501

Автоматично търси:
- `results/ews/monitoring_log.csv`
- `results/ensemble_results.csv`
- `data/processed/real_features.csv`

---

## 3. app.py — Flask уеб сървър
```bash
pip install flask pandas
python app.py
```
Отваря се на: http://localhost:5000

Поддържа:
- Upload на CSV от браузъра
- REST API: /api/data, /api/upload, /api/export
- SHAP waterfall при всеки адрес
- Мониторинг лог

---

## Формат на CSV файла
```
address,score,if_score,rf_score,gs_score,level,type,contract_rate,out_in_ratio,...
0x905b63...,0.8409,0.72,0.91,0.80,КРИТИЧЕН,Mixer,0.997,4.21,...
```
Задължителна колона: `score` (или `ensemble_score`, `risk_score`).

---

## Поставяне в проекта
```
VSCODE_PROJECT 2/
├── .venv/
├── .vscode/
├── data/
├── results/
│   └── ews/
│       └── monitoring_log.csv   ← генерира се от 07_early_warning.py
├── src/
│   ├── 00_collect_extended.py
│   ├── 00_collect_real_data.py
│   ├── 01_data_preparation.py
│   ├── 01b_prepare_real_data.py
│   ├── 02_isolation_forest.py
│   ├── 03_random_forest.py
│   ├── 04_graphsage.py
│   ├── 05_ensemble.py
│   ├── 06_shap_analysis.py
│   ├── 06b_shap_real_data.py
│   └── 07_early_warning.py
├── .env
├── .env.example
├── .gitignore
├── README.md                    ← оригиналният README
├── README_UI.md                 ← този файл
├── requirements.txt
├── setup.bat
├── dashboard.html               ← тук
├── streamlit_app.py             ← тук
└── app.py                       ← тук
```

---

## Стартиране стъпка по стъпка

### dashboard.html
1. Свали файла
2. Постави го в root-а на проекта
3. Двойно кликване → отваря се в браузъра
4. Натисни "Demo данни" или зареди CSV

### streamlit_app.py
```bash
# В терминала на VS Code:
cd ~/DIPLOMNA/vscode_project\ 2
pip install streamlit plotly pandas
streamlit run streamlit_app.py
```

### app.py
```bash
cd ~/DIPLOMNA/vscode_project\ 2
pip install flask pandas
python app.py
# Отваря се на http://localhost:5000
```
