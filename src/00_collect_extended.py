"""
00_collect_extended.py
Разширено събиране — добавяме още адреси към съществуващите данни
"""
import os, time, requests, pandas as pd, numpy as np
from dotenv import load_dotenv
load_dotenv()

ETHERSCAN_KEY = os.getenv('ETHERSCAN_API_KEY')
BASE_URL      = "https://api.etherscan.io/v2/api"
os.makedirs('data/raw', exist_ok=True)

print("=" * 55)
print("РАЗШИРЕНО СЪБИРАНЕ НА ДАННИ")
print("=" * 55)

# Зареждаме съществуващите данни
existing_file = 'data/raw/real_transactions.csv'
if os.path.exists(existing_file):
    existing = pd.read_csv(existing_file)
    existing_addrs = set(existing['address'].str.lower())
    print(f"Съществуващи адреси: {len(existing)}")
else:
    existing = pd.DataFrame()
    existing_addrs = set()

# ── НОВИ аномални адреси ───────────────────────────────────────────────
NEW_ANOMALY = [
    # Повече Tornado Cash pool адреси
    "0x12D66f87A04A9E220C9D5078658B8D1d26F7b941",  # TC 0.1 ETH
    "0x47CE0C6eD5B0Ce3d3A51fdb1C52DC66a7c3c2936",  # TC 10 ETH (v2)
    "0x910Cbd523D972eb0a6f4cAe4618aD62622b39DbF",  # TC 100 ETH (v2)
    "0xA160cdAB225685dA1d56aa342Ad8841c3b53f291",  # TC 1000 ETH
    "0x23773E65ed146A459667AD352728132c74E2b5CE",  # TC ERC20
    "0x58E8dCC13BE9780fC37158324925d6F3d32D6240",  # TC DAI 100
    "0xD691F27f38B395864Ea86CfC7253d0f8E71c1E32",  # TC DAI 1000
    "0xaEaaC358560e11f52454D997AAFF2c5731B6f8a6",  # TC DAI 10000
    "0x1E34A77868E19A6647b1f2F47B51ed72dEDE95DD",  # TC cDAI 5000
    "0xdf231d99Ff8b6c6CBF4E9B9a945CBAcEF9339178",  # TC cDAI 50000
    # Lazarus Group адреси (OFAC санкционирани)
    "0x098B716B8Aaf21512996dC57EB0615e2383E2f96",
    "0xa0e1c89Ef1a489c9C7dE96311eD5Ce5D32c20E4B",
    "0x3Fbe51f8f7e93A72B36a0Eaa42C7b2E880e3eC4C",
    "0x629e7Da20197a5429d30da36E77d06CdF796b71A",
    "0xD882cFc20F52f2599D84b8e8D58C7FB62cfE344b",
    "0x901bb9583b24D97e995513C6778dc6888AB6870e",
    "0x7F367cC41522cE07553e823bf3be79A889DEBE1B",
    "0x9F4cda013E354b8fC285BF4b9A60460cEe7f7Ea9",
    # Exploit адреси
    "0x59ABf3837Fa962d6853b4Cc0a19513AA031fd32b",  # Rari Fuse exploit
    "0x87b5BA8d98d6b79d9A97f7Be42C6Fb42Dcc5e89e",  # Nomad bridge exploit
    "0xB3c839dbde6B96D37C56ee4f9DAd3390D49310Aa",  # Deus Finance exploit  
    "0x56D8B635A7C88Fd1104D23d632AF40c1e3C3bD21",  # Beanstalk exploit
    "0xfD2d3806f4a71905aDa71350e1B5a66E71729e09",  # Fei exploit
]

# ── НОВИ нормални адреси ───────────────────────────────────────────────
NEW_NORMAL = [
    # Повече борси
    "0x564286362092D8e7936f0549571a803B203aAceD",  # Binance 12
    "0x1522900b6dAF4fA99e68EaC05fFe5c4f34cA6bE",  # Binance 13
    "0x3e5a3CEB8F89B9660Ef9B2DB7Da9b56CeDdEB4F5",  # OKX hot wallet
    "0x6cC5F688a315f3dC28A7781717a9A798a59fDA7b",  # OKX cold
    "0x236F9F97e0E62388479bf9e5BA4889e46B0273C3",  # Huobi
    "0xaB5C66752a9e8167967685F1450532fB96d5d24f",  # Huobi 2
    "0x5C985E89DDe482eFE97ea9f1950aD149Eb73829B",  # Huobi 3
    "0xDc76CD25977E0a5Ae17155770273aD58648900D3",  # Huobi 4
    "0x4e5B2e1dc63F6b91cb6Cd759936495434C7e972F",  # Bybit
    "0xf89d7b9c864f589bbF53a82105107622B35EaA40",  # Bybit 2
    # DeFi протоколи
    "0x3d9819210A31b4961b30EF54bE2aeD79B9c9Cd3B",  # Compound comptroller
    "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9",  # Aave v2
    "0x398eC7346DcD622eDc5ae82352F02bE94C62d119",  # Aave v1
    "0xB53C1a33016B2DC2fF3653530bfF1848a515c8c5",  # Aave lending pool
    "0x5d3a536E4D6DbD6114cc1Ead35777bAB948E3643",  # Compound DAI
    "0x4Ddc2D193948926D02f9B1fE9e1daa0718270ED5",  # Compound ETH
    "0xCcbA0b2bc4BAbe4cbFb6bD2f1Edc2A9e86b7845f",  # Curve 3pool
    "0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7",  # Curve 3pool 2
    "0xd632f22692FaC7611d2AA1C0D552930D43CAEd3B",  # Curve FRAX
    "0x43b4FdFD4Ff969587185cDB6f0BD875c5Fc83f8c",  # Curve ALUSD
    # Известни DeFi потребители/фондове
    "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
    "0x0000000000000000000000000000000000000000",  # Burn address
    "0xdead000000000000000042069420694206942069",  # Dead address
    "0x40B38765696e3d5d8d9d834D8AaD4bB6e418E489",  # Robinhood cold
    "0x72a53cDBBcc1b9efa39c834A540550e23463AAcB",  # Crypto.com
]

def get_transactions(address, max_tx=300):
    params = {"chainid":1,"module":"account","action":"txlist","address":address,
              "startblock":0,"endblock":99999999,"sort":"desc","offset":max_tx,"page":1,"apikey":ETHERSCAN_KEY}
    try:
        r = requests.get(BASE_URL, params=params, timeout=15)
        data = r.json()
        if data.get("status") == "1" and data.get("result"):
            return data["result"]
    except: pass
    return []

def get_balance(address):
    params = {"chainid":1,"module":"account","action":"balance","address":address,"tag":"latest","apikey":ETHERSCAN_KEY}
    try:
        r = requests.get(BASE_URL, params=params, timeout=10)
        data = r.json()
        if data.get("status") == "1":
            return int(data["result"]) / 1e18
    except: pass
    return 0.0

def extract_features(address, txs, label):
    if not txs: return None
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
    n = len(txs); out_n = sum(1 for tx in txs if tx.get("from","").lower()==addr_lower); in_n = n - out_n
    intervals = []
    if len(timestamps) > 1:
        ts = sorted(timestamps); intervals = [ts[i+1]-ts[i] for i in range(len(ts)-1)]
    return {
        "address":address,"label":label,"tx_count":n,"tx_out_count":out_n,"tx_in_count":in_n,
        "out_in_ratio":out_n/max(in_n,1),"total_value_eth":sum(values),
        "avg_value_eth":float(np.mean(values)),"max_value_eth":max(values),
        "min_value_eth":min(values),"std_value_eth":float(np.std(values)),
        "median_value_eth":float(np.median(values)),"avg_gas_gwei":float(np.mean(gas_prices)),
        "max_gas_gwei":max(gas_prices),"std_gas_gwei":float(np.std(gas_prices)),
        "error_rate":float(np.mean(errors)),"unique_counterparts":len(counterparts),
        "contract_rate":contracts/n,"avg_interval_sec":float(np.mean(intervals)) if intervals else 0,
        "min_interval_sec":min(intervals) if intervals else 0,
        "std_interval_sec":float(np.std(intervals)) if intervals else 0,
        "activity_span_days":(max(timestamps)-min(timestamps))/86400 if len(timestamps)>1 else 0,
        "balance_eth":get_balance(address),"value_concentration":max(values)/max(sum(values),1e-9),
    }

new_features = []

print(f"\n[1/3] Нови аномални адреси ({len(NEW_ANOMALY)})...")
for i, addr in enumerate(NEW_ANOMALY):
    if addr.lower() in existing_addrs:
        print(f"  {i+1:2d}: пропусната (вече съществува)")
        continue
    print(f"  {i+1:2d}/{len(NEW_ANOMALY)}: {addr[:14]}...", end=" ", flush=True)
    txs = get_transactions(addr, 300)
    feat = extract_features(addr, txs, 1)
    if feat: new_features.append(feat); print(f"✓ {len(txs)} tx")
    else: print("пропусната")
    time.sleep(0.22)

print(f"\n[2/3] Нови нормални адреси ({len(NEW_NORMAL)})...")
for i, addr in enumerate(NEW_NORMAL):
    if addr.lower() in existing_addrs:
        print(f"  {i+1:2d}: пропусната (вече съществува)")
        continue
    print(f"  {i+1:2d}/{len(NEW_NORMAL)}: {addr[:14]}...", end=" ", flush=True)
    txs = get_transactions(addr, 200)
    feat = extract_features(addr, txs, 0)
    if feat: new_features.append(feat); print(f"✓ {len(txs)} tx")
    else: print("пропусната")
    time.sleep(0.22)

print(f"\n[3/3] Още случайни адреси от последни блокове...")
try:
    r = requests.get(BASE_URL, params={"chainid":1,"module":"proxy","action":"eth_blockNumber","apikey":ETHERSCAN_KEY}, timeout=10)
    block_num = int(r.json()["result"], 16)
    random_addrs = set()
    for b in range(block_num-15, block_num):
        r2 = requests.get(BASE_URL, params={"chainid":1,"module":"proxy","action":"eth_getBlockByNumber","tag":hex(b),"boolean":"true","apikey":ETHERSCAN_KEY}, timeout=15)
        for tx in r2.json().get("result",{}).get("transactions",[])[:25]:
            if tx.get("from"): random_addrs.add(tx["from"])
        time.sleep(0.2)
    random_addrs = [a for a in random_addrs if a.lower() not in existing_addrs][:150]
    print(f"  Намерени {len(random_addrs)} нови адреса")
    collected = 0
    for i, addr in enumerate(random_addrs):
        txs = get_transactions(addr, 100)
        feat = extract_features(addr, txs, 0)
        if feat and feat["tx_count"] >= 5:
            new_features.append(feat); collected += 1
        time.sleep(0.2)
        if (i+1) % 30 == 0: print(f"  {i+1}/{len(random_addrs)} обработени, добавени: {collected}")
except Exception as e:
    print(f"  Грешка: {e}")

# Обединяване
new_df = pd.DataFrame(new_features)
if len(existing) > 0 and len(new_df) > 0:
    combined = pd.concat([existing, new_df], ignore_index=True)
elif len(new_df) > 0:
    combined = new_df
else:
    combined = existing

combined = combined.drop_duplicates(subset='address').reset_index(drop=True)
combined.to_csv('data/raw/real_transactions.csv', index=False)

print(f"\n{'='*55}")
print(f"РЕЗУЛТАТ:")
print(f"  Предишни адреси: {len(existing)}")
print(f"  Нови добавени:   {len(new_df)}")
print(f"  Общо сега:       {len(combined)}")
print(f"  Аномалии: {combined['label'].sum()} ({combined['label'].mean()*100:.1f}%)")
print(f"  Нормални: {(combined['label']==0).sum()}")
print(f"\nСледваща стъпка: python src/01b_prepare_real_data.py")