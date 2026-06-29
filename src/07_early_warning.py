"""
07_early_warning.py
Early Warning System — мониторинг на всички адреси от real_transactions.csv
"""

import os
import time
import requests
import numpy as np
import joblib
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

ETHERSCAN_KEY = os.getenv('ETHERSCAN_API_KEY')
BASE_URL      = "https://api.etherscan.io/v2/api"

os.makedirs('results/ews', exist_ok=True)

print("=" * 60)
print("EARLY WARNING SYSTEM — Мониторинг на Ethereum адреси")
print("=" * 60)

# ============================================================
# ЗАРЕЖДАНЕ НА МОДЕЛИТЕ
# ============================================================
try:
    clf = joblib.load('data/processed/random_forest.pkl')
    if_model = joblib.load('data/processed/isolation_forest.pkl')
    scaler = joblib.load('data/processed/scaler_real.pkl')
    feature_cols = joblib.load('data/processed/feature_cols_real.pkl')
    print(f"✅ Модели заредени успешно")
    print(f"   Характеристики: {len(feature_cols)}")
except Exception as e:
    print(f"❌ Грешка при зареждане на модели: {e}")
    print("   Моля, първо изпълнете: python src/05_ensemble.py")
    exit(1)

# ============================================================
# КОНСТАНТИ
# ============================================================
RISK_LOW    = 0.30
RISK_MEDIUM = 0.60
RISK_HIGH   = 0.80
WEIGHTS = {'isolation_forest': 0.20, 'random_forest': 0.50, 'graphsage': 0.30}

def risk_level(score):
    if score >= RISK_HIGH:
        return "КРИТИЧЕН"
    elif score >= RISK_MEDIUM:
        return "ВИСОК"
    elif score >= RISK_LOW:
        return "СРЕДЕН"
    else:
        return "НИСЪК"

# ============================================================
# API ФУНКЦИИ
# ============================================================
def get_transactions(address, max_tx=200):
    params = {
        "chainid": 1, "module": "account", "action": "txlist",
        "address": address, "startblock": 0, "endblock": 99999999,
        "sort": "desc", "offset": max_tx, "page": 1, "apikey": ETHERSCAN_KEY,
    }
    try:
        r = requests.get(BASE_URL, params=params, timeout=15)
        data = r.json()
        if data.get("status") == "1":
            return data["result"]
    except Exception as e:
        print(f"  API грешка: {e}")
    return []

def get_balance(address):
    params = {"chainid":1,"module":"account","action":"balance",
              "address":address,"tag":"latest","apikey":ETHERSCAN_KEY}
    try:
        r = requests.get(BASE_URL, params=params, timeout=10)
        data = r.json()
        if data.get("status") == "1":
            return int(data["result"]) / 1e18
    except:
        pass
    return 0.0

# ============================================================
# ИЗВЛИЧАНЕ НА ХАРАКТЕРИСТИКИ
# ============================================================
def extract_features(address, txs):
    if not txs:
        return None
    values, gas_prices, timestamps, errors = [], [], [], []
    contracts = 0
    counterparts = set()
    addr_lower = address.lower()
    
    for tx in txs:
        try:
            values.append(int(tx.get("value",0))/1e18)
            gas_prices.append(int(tx.get("gasPrice",0))/1e9)
            timestamps.append(int(tx.get("timeStamp",0)))
            errors.append(int(tx.get("isError",0)))
            if tx.get("input","0x") not in ("0x",""):
                contracts += 1
            fr = tx.get("from","").lower()
            to = tx.get("to","").lower()
            if fr != addr_lower:
                counterparts.add(fr)
            if to and to != addr_lower:
                counterparts.add(to)
        except:
            continue
    
    if not values:
        return None
    
    n = len(txs)
    out_n = sum(1 for tx in txs if tx.get("from","").lower() == addr_lower)
    in_n = n - out_n
    
    intervals = []
    if len(timestamps) > 1:
        ts = sorted(timestamps)
        intervals = [ts[i+1]-ts[i] for i in range(len(ts)-1)]
    
    return {
        'tx_count': n,
        'tx_out_count': out_n,
        'tx_in_count': in_n,
        'out_in_ratio': out_n / max(in_n, 1),
        'total_value_eth': sum(values),
        'avg_value_eth': float(np.mean(values)),
        'max_value_eth': max(values),
        'min_value_eth': min(values),
        'std_value_eth': float(np.std(values)),
        'median_value_eth': float(np.median(values)),
        'avg_gas_gwei': float(np.mean(gas_prices)) if gas_prices else 0,
        'max_gas_gwei': max(gas_prices) if gas_prices else 0,
        'std_gas_gwei': float(np.std(gas_prices)) if gas_prices else 0,
        'error_rate': float(np.mean(errors)) if errors else 0,
        'unique_counterparts': len(counterparts),
        'contract_rate': contracts / n,
        'avg_interval_sec': float(np.mean(intervals)) if intervals else 0,
        'min_interval_sec': min(intervals) if intervals else 0,
        'std_interval_sec': float(np.std(intervals)) if intervals else 0,
        'activity_span_days': (max(timestamps)-min(timestamps))/86400 if len(timestamps)>1 else 0,
        'balance_eth': get_balance(address),
        'value_concentration': max(values) / max(sum(values), 1e-9),
    }

# ============================================================
# АНАЛИЗ НА АДРЕС
# ============================================================
def analyze_address(address):
    txs = get_transactions(address, 200)
    if not txs or len(txs) < 5:
        return None
    
    feat = extract_features(address, txs)
    if not feat:
        return None
    
    # Подготовка на вектора
    feat_vector = np.array([[feat.get(f, 0) for f in feature_cols]], dtype=np.float32)
    feat_scaled = scaler.transform(feat_vector)
    
    # 1. Isolation Forest
    if_score = -if_model.score_samples(feat_scaled)
    if_score = (if_score[0] - if_score.min()) / (if_score.max() - if_score.min() + 1e-9)
    if_score = np.clip(if_score, 0, 1)
    
    # 2. Random Forest
    rf_probs = clf.predict_proba(feat_scaled)[0]
    rf_score = rf_probs[1]
    
    # 3. Комбиниран резултат
    risk_score = (WEIGHTS['isolation_forest'] * if_score + 
                  WEIGHTS['random_forest'] * rf_score)
    
    prediction = 1 if risk_score >= 0.5 else 0
    level = risk_level(risk_score)
    
    # Лог запис
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'address': address,
        'score': round(risk_score, 4),
        'if_score': round(if_score, 4),
        'rf_score': round(rf_score, 4),
        'level': level,
        'prediction': prediction,
        **feat
    }
    
    return log_entry

# ============================================================
# ЗАРЕЖДАНЕ НА ВСИЧКИ АДРЕСИ
# ============================================================
def load_all_addresses():
    """Зарежда всички адреси от real_transactions.csv"""
    df = pd.read_csv('data/raw/real_transactions.csv')
    print(f"📊 Заредени {len(df)} адреса от real_transactions.csv")
    print(f"   Аномални: {df['label'].sum()} ({(df['label'].sum()/len(df))*100:.1f}%)")
    print(f"   Нормални: {(df['label']==0).sum()}")
    return df['address'].tolist()

# ============================================================
# ОСНОВНА ПРОГРАМА
# ============================================================
print("\n" + "=" * 60)
print("АНАЛИЗ НА ВСИЧКИ АДРЕСИ")
print("=" * 60)

all_addresses = load_all_addresses()
print(f"\nЗапочва анализ на {len(all_addresses)} адреса...")
print(f"Това може да отнеме около 2-3 минути...")

results = []
total = len(all_addresses)

for i, address in enumerate(all_addresses):
    if (i + 1) % 50 == 0:
        print(f"  Прогрес: {i+1}/{total}")
    
    log_entry = analyze_address(address)
    if log_entry:
        results.append(log_entry)
    time.sleep(0.15)  # Rate limiting

# ============================================================
# ЗАПАЗВАНЕ НА РЕЗУЛТАТИТЕ
# ============================================================
print(f"\n✅ Анализирани {len(results)} адреса")

if results:
    df_results = pd.DataFrame(results)
    
    # Запазване на CSV
    output_file = 'results/ews/monitoring_log.csv'
    df_results.to_csv(output_file, index=False)
    print(f"📁 Резултатите са запазени в: {output_file}")
    
    # Статистика
    critical = len(df_results[df_results['level'] == 'КРИТИЧЕН'])
    high = len(df_results[df_results['level'] == 'ВИСОК'])
    medium = len(df_results[df_results['level'] == 'СРЕДЕН'])
    low = len(df_results[df_results['level'] == 'НИСЪК'])
    
    print("\n" + "=" * 60)
    print("ОБОБЩЕНИЕ НА РЕЗУЛТАТИТЕ")
    print("=" * 60)
    print(f"  Общо адреси:      {len(df_results)}")
    print(f"  КРИТИЧЕН:         {critical} ({(critical/len(df_results))*100:.1f}%)")
    print(f"  ВИСОК:            {high} ({(high/len(df_results))*100:.1f}%)")
    print(f"  СРЕДЕН:           {medium} ({(medium/len(df_results))*100:.1f}%)")
    print(f"  НИСЪК:            {low} ({(low/len(df_results))*100:.1f}%)")
    
    # Топ 5 най-рискови
    print("\n🏆 Топ 5 най-рискови адреса:")
    top5 = df_results.nlargest(5, 'score')[['address', 'score', 'level']]
    for _, row in top5.iterrows():
        print(f"  {row['address']}... {row['score']:.4f}  {row['level']}")
    
    print("\n✅ Early Warning System завърши успешно!")
    print("📊 Стартирайте: streamlit run streamlit_app.py")
else:
    print("❌ Няма анализирани адреси")