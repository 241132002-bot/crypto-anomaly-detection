"""
07_early_warning.py
Early Warning System — мониторинг на адреси в реално време
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

# Зареждане на модела
clf     = joblib.load('results/random_forest_model.pkl')
scaler  = joblib.load('data/processed/scaler.pkl')
f_names = joblib.load('data/processed/feature_names.pkl')

print(f"Модел зареден: Random Forest")
print(f"Характеристики: {len(f_names)}")

# ── Прагове за риск ───────────────────────────────────────────────────
RISK_LOW    = 0.30
RISK_MEDIUM = 0.60
RISK_HIGH   = 0.80

def risk_level(score):
    if score >= RISK_HIGH:
        return "🔴 КРИТИЧЕН"
    elif score >= RISK_MEDIUM:
        return "🟠 ВИСОК"
    elif score >= RISK_LOW:
        return "🟡 СРЕДЕН"
    else:
        return "🟢 НИСъК"

# ── API функции ───────────────────────────────────────────────────────
def get_transactions(address, max_tx=200):
    params = {
        "chainid": 1, "module": "account", "action": "txlist",
        "address": address, "startblock": 0, "endblock": 99999999,
        "sort": "desc", "offset": max_tx, "page": 1, "apikey": ETHERSCAN_KEY,
    }
    try:
        r    = requests.get(BASE_URL, params=params, timeout=15)
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

def extract_features(address, txs):
    if not txs:
        return None
    values, gas_prices, timestamps, errors = [], [], [], []
    contracts = 0; counterparts = set(); addr_lower = address.lower()
    for tx in txs:
        try:
            values.append(int(tx.get("value",0))/1e18)
            gas_prices.append(int(tx.get("gasPrice",0))/1e9)
            timestamps.append(int(tx.get("timeStamp",0)))
            errors.append(int(tx.get("isError",0)))
            if tx.get("input","0x") not in ("0x",""): contracts += 1
            fr = tx.get("from","").lower(); to = tx.get("to","").lower()
            if fr != addr_lower: counterparts.add(fr)
            if to and to != addr_lower: counterparts.add(to)
        except: continue
    if not values: return None
    n = len(txs)
    out_n = sum(1 for tx in txs if tx.get("from","").lower()==addr_lower)
    in_n  = n - out_n
    intervals = []
    if len(timestamps) > 1:
        ts = sorted(timestamps); intervals = [ts[i+1]-ts[i] for i in range(len(ts)-1)]
    return {
        'tx_count': n, 'tx_out_count': out_n, 'tx_in_count': in_n,
        'out_in_ratio': out_n/max(in_n,1), 'total_value_eth': sum(values),
        'avg_value_eth': float(np.mean(values)), 'max_value_eth': max(values),
        'min_value_eth': min(values), 'std_value_eth': float(np.std(values)),
        'median_value_eth': float(np.median(values)),
        'avg_gas_gwei': float(np.mean(gas_prices)), 'max_gas_gwei': max(gas_prices),
        'std_gas_gwei': float(np.std(gas_prices)), 'error_rate': float(np.mean(errors)),
        'unique_counterparts': len(counterparts), 'contract_rate': contracts/n,
        'avg_interval_sec': float(np.mean(intervals)) if intervals else 0,
        'min_interval_sec': min(intervals) if intervals else 0,
        'std_interval_sec': float(np.std(intervals)) if intervals else 0,
        'activity_span_days': (max(timestamps)-min(timestamps))/86400 if len(timestamps)>1 else 0,
        'balance_eth': get_balance(address),
        'value_concentration': max(values)/max(sum(values),1e-9),
    }

def analyze_address(address):
    print(f"\n{'─'*60}")
    print(f"Анализ на адрес: {address}")
    print(f"Час: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'─'*60}")

    txs = get_transactions(address, 300)
    if not txs:
        print("  ⚠️  Няма транзакции или адресът не е намерен")
        return None

    print(f"  Намерени транзакции: {len(txs)}")
    feat = extract_features(address, txs)
    if not feat:
        print("  ⚠️  Не могат да се извлекат характеристики")
        return None

    # Подреждане на характеристиките в правилния ред
    feat_vector = np.array([[feat.get(f, 0) for f in f_names]], dtype=np.float32)
    feat_scaled = scaler.transform(feat_vector)

    # Прогноза
    risk_score = clf.predict_proba(feat_scaled)[0][1]
    prediction = clf.predict(feat_scaled)[0]
    level      = risk_level(risk_score)

    print(f"\n  📊 РЕЗУЛТАТ:")
    print(f"  Risk Score:  {risk_score:.4f} ({risk_score*100:.1f}%)")
    print(f"  Ниво:        {level}")
    print(f"  Класификация: {'⚠️  АНОМАЛИЯ' if prediction == 1 else '✅ НОРМАЛЕН'}")

    print(f"\n  📈 КЛЮЧОВИ ХАРАКТЕРИСТИКИ:")
    print(f"  Брой транзакции:      {feat['tx_count']}")
    print(f"  Уникални контрагенти: {feat['unique_counterparts']}")
    print(f"  Общ обем (ETH):       {feat['total_value_eth']:.4f}")
    print(f"  Макс транзакция:      {feat['max_value_eth']:.4f} ETH")
    print(f"  Дял договори:         {feat['contract_rate']*100:.1f}%")
    print(f"  Баланс:               {feat['balance_eth']:.4f} ETH")
    print(f"  Активност (дни):      {feat['activity_span_days']:.0f}")

    if risk_score >= RISK_MEDIUM:
        print(f"\n  🚨 ПРЕДУПРЕЖДЕНИЕ:")
        if feat['contract_rate'] > 0.7:
            print(f"  - Висок дял на договорни взаимодействия ({feat['contract_rate']*100:.0f}%)")
        if feat['value_concentration'] > 0.8:
            print(f"  - Концентрирани транзакции (value_concentration={feat['value_concentration']:.2f})")
        if feat['avg_interval_sec'] < 60:
            print(f"  - Много кратки интервали между транзакции ({feat['avg_interval_sec']:.0f} сек)")
        if feat['out_in_ratio'] > 10:
            print(f"  - Нетипично висок out/in ratio ({feat['out_in_ratio']:.1f})")

    # Запазване на лог
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'address': address,
        'risk_score': risk_score,
        'level': level,
        'prediction': prediction,
        **feat
    }

    log_file = 'results/ews/monitoring_log.csv'
    log_df   = pd.DataFrame([log_entry])
    if os.path.exists(log_file):
        log_df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_file, index=False)

    return risk_score

# ── Демонстрация с тестови адреси ─────────────────────────────────────
print("\nДЕМОНСТРАЦИЯ НА EARLY WARNING SYSTEM")
print("="*60)

TEST_ADDRESSES = [
    # Известен Tornado Cash адрес (очакваме висок риск)
    ("0x722122dF12D4e14e13Ac3b6895a86e84145b6967", "Tornado Cash Router"),
    # Binance борса (очакваме нисък риск)
    ("0x28C6c06298d514Db089934071355E5743bf21d60", "Binance Exchange"),
    # Втори Tornado Cash
    ("0x910Cbd523D972eb0a6f4cAe4618aD62622b39DbF", "Tornado Cash 100 ETH"),
    # Kraken борса
    ("0x2910543Af39abA0Cd09dBb2D50200b3E800A63D2", "Kraken Exchange"),
]

results = []
for address, label in TEST_ADDRESSES:
    print(f"\n[Тест] {label}")
    score = analyze_address(address)
    if score is not None:
        results.append((label, address, score, risk_level(score)))
    time.sleep(0.5)

# ── Обобщение ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("ОБОБЩЕНИЕ НА МОНИТОРИНГА")
print(f"{'='*60}")
print(f"{'Адрес':<35} {'Score':>7} {'Ниво'}")
print("-"*60)
for label, addr, score, level in results:
    print(f"{label:<35} {score:>7.4f}  {level}")

print(f"\nЛогът е запазен: results/ews/monitoring_log.csv")
print("\n✅ Early Warning System демонстрацията е завършена!")
print("\nЗа мониторинг на собствен адрес добави го в TEST_ADDRESSES")
