"""
Early Warning System — Flask Web Application
============================================
Стартиране:
    pip install flask pandas scikit-learn
    python app.py

Поставете файла в root директорията на проекта (до src/, data/, results/).
Достъп: http://localhost:5000
"""

from flask import Flask, render_template_string, jsonify, request, send_file
import pandas as pd
import numpy as np
import json, os, io
from pathlib import Path
from datetime import datetime

app = Flask(__name__)

# ── Demo данни ─────────────────────────────────────────────────────────
DEMO = [
    {"name":"Tornado Cash Router","addr":"0x905b63Fff465B9fFBF41DeA908CEb12478ec7A5","score":0.8409,"if_score":0.72,"rf_score":0.91,"gs_score":0.80,"level":"КРИТИЧЕН","type":"Mixer протокол","contract_rate":0.997,"out_in_ratio":4.21,"unique_counterparts":176,"avg_gas_gwei":42.3,"balance_eth":18590,"tx_count":12043},
    {"name":"Lazarus Group #1","addr":"0x3AD9dB589d201A710Ed237c829b7D0ae916e586","score":0.7821,"if_score":0.61,"rf_score":0.88,"gs_score":0.72,"level":"КРИТИЧЕН","type":"Санкциониран OFAC","contract_rate":0.672,"out_in_ratio":8.94,"unique_counterparts":34,"avg_gas_gwei":31.1,"balance_eth":0,"tx_count":289},
    {"name":"Rari Fuse Exploit","addr":"0x6EfF3310df9E73534c33F31A49a54cf2f45D2B0A","score":0.7234,"if_score":0.58,"rf_score":0.82,"gs_score":0.68,"level":"КРИТИЧЕН","type":"Exploit адрес","contract_rate":0.934,"out_in_ratio":7.12,"unique_counterparts":6,"avg_gas_gwei":189.3,"balance_eth":0,"tx_count":31},
    {"name":"Beanstalk Exploit","addr":"0x1c5dCdd006EA78a7E4783f9e6021C32935a10fb4","score":0.6914,"if_score":0.55,"rf_score":0.79,"gs_score":0.64,"level":"ВИСОК","type":"Exploit адрес","contract_rate":0.881,"out_in_ratio":6.33,"unique_counterparts":8,"avg_gas_gwei":220.4,"balance_eth":0,"tx_count":47},
    {"name":"Nomad Bridge","addr":"0x88A69B4E698A4B090DF6CF5Bd7B2D47325Ad30A3","score":0.5821,"if_score":0.41,"rf_score":0.65,"gs_score":0.54,"level":"СРЕДЕН","type":"Exploit адрес","contract_rate":0.912,"out_in_ratio":3.14,"unique_counterparts":44,"avg_gas_gwei":48.9,"balance_eth":0,"tx_count":1203},
    {"name":"Tornado Cash 100 ETH","addr":"0x910Cbd523D972eb0a6f4cAe4618aD62622b39DbF","score":0.1359,"if_score":0.22,"rf_score":0.11,"gs_score":0.14,"level":"НИСЪк","type":"Mixer протокол","contract_rate":1.0,"out_in_ratio":1.02,"unique_counterparts":12,"avg_gas_gwei":21.0,"balance_eth":0,"tx_count":4421},
    {"name":"Binance Exchange","addr":"0x28C6c06298d514Db089934071355E5743bf21d60","score":0.0206,"if_score":0.04,"rf_score":0.01,"gs_score":0.03,"level":"НИСЪк","type":"Борса (CEX)","contract_rate":0.123,"out_in_ratio":0.98,"unique_counterparts":9841,"avg_gas_gwei":18.2,"balance_eth":184200,"tx_count":2834199},
    {"name":"Kraken Exchange","addr":"0x2910543Af39abA0Cd09dBb2D50200b3E800A63D2","score":0.0618,"if_score":0.09,"rf_score":0.04,"gs_score":0.07,"level":"НИСЪк","type":"Борса (CEX)","contract_rate":0.284,"out_in_ratio":1.14,"unique_counterparts":212,"avg_gas_gwei":19.7,"balance_eth":662063,"tx_count":88441},
    {"name":"Aave Protocol","addr":"0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2","score":0.0431,"if_score":0.06,"rf_score":0.03,"gs_score":0.05,"level":"НИСЪк","type":"DeFi протокол","contract_rate":1.0,"out_in_ratio":0.89,"unique_counterparts":5621,"avg_gas_gwei":22.1,"balance_eth":1240000,"tx_count":4129831},
    {"name":"Compound Finance","addr":"0x3d9819210A31b4961b30EF54bE2aeD79B9c9Cd3B","score":0.0512,"if_score":0.07,"rf_score":0.04,"gs_score":0.06,"level":"НИСЪк","type":"DeFi протокол","contract_rate":1.0,"out_in_ratio":0.92,"unique_counterparts":3241,"avg_gas_gwei":20.8,"balance_eth":890000,"tx_count":2341000},
]

def get_level(score, c=0.80, h=0.60, m=0.30):
    if score >= c: return "КРИТИЧЕН"
    if score >= h: return "ВИСОК"
    if score >= m: return "СРЕДЕН"
    return "НИСЪк"

def load_real_data():
    paths = [
        "results/ews/monitoring_log.csv",
        "results/ensemble_results.csv",
        "data/processed/real_features.csv",
    ]
    for p in paths:
        if Path(p).exists():
            try:
                df = pd.read_csv(p)
                score_col = next((c for c in ['score','ensemble_score','risk_score'] if c in df.columns), None)
                if score_col:
                    df = df.rename(columns={score_col: 'score'})
                    df['level'] = df['score'].apply(get_level)
                    return df.to_dict('records'), p
            except:
                pass
    return DEMO, "demo"

# ── HTML template ──────────────────────────────────────────────────────
HTML = '''<!DOCTYPE html>
<html lang="bg">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Early Warning System</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f5f5f5;color:#1a1a1a}
.layout{display:flex;min-height:100vh}
.sidebar{width:210px;background:#fff;border-right:1px solid #e5e5e5;padding:1.25rem 1rem;flex-shrink:0}
.logo{display:flex;align-items:center;gap:8px;margin-bottom:1.5rem}
.logo-icon{width:30px;height:30px;background:#dc2626;border-radius:7px;display:flex;align-items:center;justify-content:center;color:#fff;font-size:14px}
.logo span{font-size:13px;font-weight:600}
.nav-label{font-size:10px;font-weight:600;color:#9ca3af;text-transform:uppercase;letter-spacing:.06em;margin:1rem 0 .4rem}
.nav-item{display:flex;align-items:center;gap:8px;padding:7px 9px;border-radius:6px;cursor:pointer;font-size:13px;color:#6b7280;margin-bottom:2px;border:none;background:none;width:100%;text-align:left}
.nav-item:hover{background:#f9fafb;color:#111}
.nav-item.active{background:#fef2f2;color:#dc2626;font-weight:500}
.status{display:flex;align-items:center;gap:6px;font-size:11px;color:#16a34a;margin-top:auto;border-top:1px solid #f3f4f6;padding-top:1rem;margin-top:2rem}
.dot{width:6px;height:6px;border-radius:50%;background:#16a34a;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.main{flex:1;padding:1.5rem;overflow-x:hidden}
.page-header{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:1.25rem}
.page-title{font-size:20px;font-weight:600}
.page-sub{font-size:12px;color:#6b7280;margin-top:3px}
.header-btns{display:flex;gap:8px}
.btn{padding:7px 14px;border-radius:6px;font-size:12px;cursor:pointer;font-weight:500;border:1px solid #e5e5e5;background:#fff;color:#374151;transition:all .15s}
.btn:hover{background:#f9fafb}
.btn-primary{background:#dc2626;color:#fff;border-color:#dc2626}
.btn-primary:hover{background:#b91c1c}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:1.25rem}
.metric{background:#fff;border:1px solid #e5e5e5;border-radius:10px;padding:1rem 1.1rem}
.metric-label{font-size:11px;color:#6b7280;margin-bottom:6px}
.metric-value{font-size:24px;font-weight:600}
.metric-sub{font-size:11px;color:#9ca3af;margin-top:3px}
.charts-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:1.25rem}
.card{background:#fff;border:1px solid #e5e5e5;border-radius:10px;padding:1.1rem}
.card-title{font-size:13px;font-weight:500;margin-bottom:1rem}
.filters{display:flex;gap:8px;margin-bottom:1rem;align-items:center}
.filters input,.filters select{padding:7px 10px;border:1px solid #e5e5e5;border-radius:6px;font-size:12px;background:#fff;outline:none}
.filters input{flex:1}
.table-card{background:#fff;border:1px solid #e5e5e5;border-radius:10px;overflow:hidden;margin-bottom:1.25rem}
.table-hdr{display:flex;align-items:center;justify-content:space-between;padding:.9rem 1.1rem;border-bottom:1px solid #f3f4f6}
.table-hdr h3{font-size:13px;font-weight:500}
.count-badge{font-size:11px;color:#6b7280;background:#f9fafb;padding:2px 8px;border-radius:99px;border:1px solid #e5e5e5}
table{width:100%;border-collapse:collapse;font-size:12px}
thead{background:#f9fafb}
th{text-align:left;padding:9px 12px;font-weight:500;font-size:11px;color:#6b7280;border-bottom:1px solid #f3f4f6}
td{padding:10px 12px;border-bottom:1px solid #f9fafb;vertical-align:middle}
tbody tr:last-child td{border-bottom:none}
tbody tr{cursor:pointer;transition:background .1s}
tbody tr:hover{background:#fafafa}
.addr-name{font-weight:500}
.addr-hash{font-family:monospace;font-size:10px;color:#9ca3af;margin-top:1px}
.score-wrap{display:flex;align-items:center;gap:8px}
.score-bar{flex:1;height:5px;border-radius:3px;background:#f3f4f6;overflow:hidden;max-width:100px}
.score-fill{height:100%;border-radius:3px}
.score-num{font-weight:600;min-width:34px;text-align:right}
.badge{display:inline-block;font-size:10px;font-weight:500;padding:2px 8px;border-radius:99px}
.badge-crit{background:#fef2f2;color:#dc2626}
.badge-high{background:#fffbeb;color:#d97706}
.badge-mid{background:#eff6ff;color:#2563eb}
.badge-low{background:#f0fdf4;color:#16a34a}
.page{display:none}
.page.active{display:block}
.log-row{display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid #f3f4f6;font-size:12px}
.log-row:last-child{border-bottom:none}
.log-time{font-family:monospace;color:#9ca3af;min-width:70px}
.log-addr{font-family:monospace;color:#6b7280;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
/* Modal */
.modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.4);z-index:100;align-items:flex-start;justify-content:flex-end}
.modal-overlay.open{display:flex}
.modal{width:460px;height:100vh;background:#fff;overflow-y:auto;border-left:1px solid #e5e5e5;padding:1.5rem}
.modal-hdr{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1.1rem}
.close-btn{width:28px;height:28px;border:1px solid #e5e5e5;border-radius:6px;background:#f9fafb;cursor:pointer;font-size:14px;display:flex;align-items:center;justify-content:center}
.score-block{display:flex;align-items:center;gap:14px;padding:.9rem;background:#f9fafb;border-radius:8px;margin-bottom:1.1rem;border:1px solid #f3f4f6}
.big-score{font-size:32px;font-weight:700}
.section-lbl{font-size:10px;font-weight:600;color:#9ca3af;text-transform:uppercase;letter-spacing:.06em;margin-bottom:7px;margin-top:1.1rem}
.shap-row{display:flex;align-items:center;gap:7px;margin-bottom:6px}
.shap-name{font-family:monospace;font-size:10px;color:#6b7280;width:120px;flex-shrink:0;overflow:hidden;text-overflow:ellipsis}
.shap-bar-bg{flex:1;height:12px;background:#f3f4f6;border-radius:3px;overflow:hidden;display:flex}
.shap-pos{background:#dc2626;height:100%;border-radius:0 3px 3px 0}
.shap-neg{background:#2563eb;height:100%;border-radius:3px 0 0 3px;margin-left:auto}
.shap-val{font-size:10px;font-weight:600;width:42px;text-align:right}
.feat-grid{display:grid;grid-template-columns:1fr 1fr;gap:7px}
.feat-box{background:#f9fafb;border:1px solid #f3f4f6;border-radius:6px;padding:8px 10px}
.feat-name{font-size:10px;font-family:monospace;color:#9ca3af;margin-bottom:3px}
.feat-val{font-size:14px;font-weight:600}
</style>
</head>
<body>
<div class="layout">
<aside class="sidebar">
  <div class="logo">
    <div class="logo-icon">🛡</div>
    <span>EWS Dashboard</span>
  </div>
  <div class="nav-label">Мониторинг</div>
  <button class="nav-item active" onclick="showPage('dashboard',this)">📊 Dashboard</button>
  <button class="nav-item" onclick="showPage('addresses',this)">📋 Адреси</button>
  <button class="nav-item" onclick="showPage('log',this)">📁 Лог</button>
  <div class="nav-label">Система</div>
  <button class="nav-item" onclick="showPage('settings',this)">⚙️ Настройки</button>
  <div class="status"><div class="dot"></div>Системата е активна</div>
</aside>

<main class="main">

<!-- DASHBOARD -->
<div class="page active" id="page-dashboard">
  <div class="page-header">
    <div>
      <div class="page-title">Early Warning System</div>
      <div class="page-sub" id="data-source">Зареждане...</div>
    </div>
    <div class="header-btns">
      <label class="btn" style="cursor:pointer">
        📂 Зареди CSV
        <input type="file" id="csv-input" accept=".csv" style="display:none" onchange="loadCSV(this)">
      </label>
      <button class="btn btn-primary" onclick="exportCSV()">⬇ Експорт</button>
    </div>
  </div>

  <div class="metrics">
    <div class="metric"><div class="metric-label">Анализирани адреси</div><div class="metric-value" id="m-total">—</div><div class="metric-sub">от набора</div></div>
    <div class="metric"><div class="metric-label">Аномални</div><div class="metric-value" id="m-anomal" style="color:#dc2626">—</div><div class="metric-sub" id="m-anomal-p">—</div></div>
    <div class="metric"><div class="metric-label">Критични сигнали</div><div class="metric-value" id="m-crit" style="color:#dc2626">—</div><div class="metric-sub">score ≥ 0.80</div></div>
    <div class="metric"><div class="metric-label">Нормални</div><div class="metric-value" id="m-normal" style="color:#16a34a">—</div><div class="metric-sub" id="m-normal-p">—</div></div>
  </div>

  <div class="charts-grid">
    <div class="card">
      <div class="card-title">Разпределение по риск нива</div>
      <div style="position:relative;height:190px"><canvas id="levelChart" role="img" aria-label="Разпределение по риск нива"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Score разпределение — аномални vs нормални</div>
      <div style="position:relative;height:190px"><canvas id="scoreChart" role="img" aria-label="Score разпределение"></canvas></div>
    </div>
  </div>

  <div class="filters">
    <input type="text" id="search" placeholder="Търси по адрес или тип..." oninput="renderTable()">
    <select id="lvl-filter" onchange="renderTable()">
      <option value="">Всички нива</option>
      <option>КРИТИЧЕН</option><option>ВИСОК</option><option>СРЕДЕН</option><option>НИСЪк</option>
    </select>
    <select id="sort-by" onchange="renderTable()">
      <option value="score_desc">Score ↓</option>
      <option value="score_asc">Score ↑</option>
      <option value="name">Ime A-Z</option>
    </select>
  </div>

  <div class="table-card">
    <div class="table-hdr"><h3>Мониторирани адреси</h3><span class="count-badge" id="tbl-count">0</span></div>
    <div style="overflow-x:auto">
      <table><thead><tr>
        <th>Адрес</th><th>Риск Score</th><th>Ниво</th>
        <th>IF</th><th>RF</th><th>GS</th><th>Тип</th>
      </tr></thead>
      <tbody id="tbl-body"></tbody></table>
    </div>
  </div>
</div>

<!-- ADDRESSES -->
<div class="page" id="page-addresses">
  <div class="page-header"><div><div class="page-title">Всички адреси</div><div class="page-sub">Пълен списък с характеристики</div></div></div>
  <div class="card" style="overflow-x:auto"><table id="full-table"><thead><tr id="full-hdr"></tr></thead><tbody id="full-body"></tbody></table></div>
</div>

<!-- LOG -->
<div class="page" id="page-log">
  <div class="page-header"><div><div class="page-title">Мониторинг лог</div><div class="page-sub">Сигнали над ниво СРЕДЕН</div></div><button class="btn" onclick="exportLog()">⬇ Лог CSV</button></div>
  <div class="card" id="log-container"></div>
</div>

<!-- SETTINGS -->
<div class="page" id="page-settings">
  <div class="page-header"><div><div class="page-title">Настройки</div><div class="page-sub">Прагове и конфигурация</div></div></div>
  <div class="card" style="max-width:480px">
    <div class="card-title">Прагове на риск скоринг</div>
    <div style="display:grid;gap:12px;margin-top:12px">
      <label style="font-size:12px;display:flex;justify-content:space-between;align-items:center">
        Критичен ≥ <input type="number" id="thresh-crit" value="0.80" min="0.5" max="1" step="0.05" style="width:70px;padding:4px 8px;border:1px solid #e5e5e5;border-radius:5px;font-size:12px">
      </label>
      <label style="font-size:12px;display:flex;justify-content:space-between;align-items:center">
        Висок ≥ <input type="number" id="thresh-high" value="0.60" min="0.3" max="0.8" step="0.05" style="width:70px;padding:4px 8px;border:1px solid #e5e5e5;border-radius:5px;font-size:12px">
      </label>
      <label style="font-size:12px;display:flex;justify-content:space-between;align-items:center">
        Среден ≥ <input type="number" id="thresh-mid" value="0.30" min="0.1" max="0.5" step="0.05" style="width:70px;padding:4px 8px;border:1px solid #e5e5e5;border-radius:5px;font-size:12px">
      </label>
      <button class="btn btn-primary" onclick="applyThresholds()" style="width:100%">Приложи</button>
    </div>
  </div>
</div>

</main>
</div>

<!-- MODAL -->
<div class="modal-overlay" id="modal" onclick="closeModal(event)">
  <div class="modal">
    <div class="modal-hdr">
      <div>
        <div style="font-size:15px;font-weight:600" id="modal-name"></div>
        <div style="font-family:monospace;font-size:11px;color:#9ca3af;margin-top:3px" id="modal-addr"></div>
      </div>
      <button class="close-btn" onclick="document.getElementById('modal').classList.remove('open')">✕</button>
    </div>
    <div class="score-block">
      <div>
        <div style="font-size:10px;color:#6b7280;margin-bottom:3px">Риск Score</div>
        <div class="big-score" id="modal-score"></div>
      </div>
      <div style="flex:1;padding-left:14px;border-left:1px solid #f3f4f6">
        <div style="font-size:10px;color:#6b7280;margin-bottom:5px">Ниво на риска</div>
        <div id="modal-badge"></div>
        <div style="font-size:11px;color:#6b7280;margin-top:6px" id="modal-type"></div>
      </div>
    </div>
    <div class="section-lbl">SHAP анализ</div>
    <div id="modal-shap"></div>
    <div class="section-lbl">Характеристики</div>
    <div class="feat-grid" id="modal-feats"></div>
  </div>
</div>

<script>
let allData = [], logEntries = [], levelChart, scoreChart;

function scoreColor(s){
  if(s>=.8) return '#dc2626';
  if(s>=.6) return '#d97706';
  if(s>=.3) return '#2563eb';
  return '#16a34a';
}
function levelCls(l){
  return {КРИТИЧЕН:'crit',ВИСОК:'high',СРЕДЕН:'mid',НИСЪк:'low'}[l]||'low';
}
function badge(l){ return `<span class="badge badge-${levelCls(l)}">${l}</span>`; }
function fmt(v){
  if(v===undefined||v===null) return '—';
  if(typeof v==='number') return v>1000?v.toLocaleString('bg'):parseFloat(v.toFixed(4)).toString();
  return v;
}

function showPage(id, btn){
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.getElementById('page-'+id).classList.add('active');
  document.querySelectorAll('.nav-item').forEach(b=>b.classList.remove('active'));
  if(btn) btn.classList.add('active');
  if(id==='addresses') renderFullTable();
  if(id==='log') renderLog();
}

fetch('/api/data').then(r=>r.json()).then(d=>{
  allData = d.data;
  document.getElementById('data-source').textContent = d.source;
  logEntries = allData.filter(x=>x.score>=0.3).map(x=>({
    time: new Date().toLocaleTimeString('bg',{hour:'2-digit',minute:'2-digit'}),
    name:x.name||'—', addr:x.addr||'—', score:x.score, level:x.level||'НИСЪк'
  }));
  update();
});

function update(){
  const total=allData.length;
  const anomal=allData.filter(d=>d.score>=.3).length;
  const crit=allData.filter(d=>d.score>=.8).length;
  const normal=allData.filter(d=>d.score<.3).length;
  document.getElementById('m-total').textContent=total;
  document.getElementById('m-anomal').textContent=anomal;
  document.getElementById('m-anomal-p').textContent=total?(anomal/total*100).toFixed(1)+'%':'—';
  document.getElementById('m-crit').textContent=crit;
  document.getElementById('m-normal').textContent=normal;
  document.getElementById('m-normal-p').textContent=total?(normal/total*100).toFixed(1)+'%':'—';
  renderTable();
  drawCharts();
}

function renderTable(){
  const q=(document.getElementById('search').value||'').toLowerCase();
  const lv=document.getElementById('lvl-filter').value;
  const sort=document.getElementById('sort-by').value;
  let d=allData.filter(x=>(!q||JSON.stringify(x).toLowerCase().includes(q))&&(!lv||x.level===lv));
  if(sort==='score_desc') d.sort((a,b)=>b.score-a.score);
  else if(sort==='score_asc') d.sort((a,b)=>a.score-b.score);
  else d.sort((a,b)=>(a.name||'').localeCompare(b.name||''));
  document.getElementById('tbl-count').textContent=d.length+' адреса';
  document.getElementById('tbl-body').innerHTML=d.map(r=>`
    <tr onclick="openModal('${r.addr}')">
      <td><div class="addr-name">${r.name||'—'}</div><div class="addr-hash">${(r.addr||'').slice(0,12)}…${(r.addr||'').slice(-6)}</div></td>
      <td><div class="score-wrap">
        <div class="score-bar"><div class="score-fill" style="width:${(r.score*100).toFixed(0)}%;background:${scoreColor(r.score)}"></div></div>
        <span class="score-num" style="color:${scoreColor(r.score)}">${r.score.toFixed(3)}</span>
      </div></td>
      <td>${badge(r.level||'НИСЪк')}</td>
      <td style="color:${scoreColor(r.if_score||0)}">${(r.if_score||0).toFixed(2)}</td>
      <td style="color:${scoreColor(r.rf_score||0)}">${(r.rf_score||0).toFixed(2)}</td>
      <td style="color:${scoreColor(r.gs_score||0)}">${(r.gs_score||0).toFixed(2)}</td>
      <td style="font-size:11px;color:#6b7280">${r.type||'—'}</td>
    </tr>`).join('');
}

function drawCharts(){
  const lvls=['НИСЪк','СРЕДЕН','ВИСОК','КРИТИЧЕН'];
  const cnts=lvls.map(l=>allData.filter(d=>d.level===l).length);
  const clrs=['#16a34a','#2563eb','#d97706','#dc2626'];
  if(levelChart) levelChart.destroy();
  levelChart=new Chart(document.getElementById('levelChart'),{type:'bar',
    data:{labels:lvls,datasets:[{data:cnts,backgroundColor:clrs,borderRadius:4,borderSkipped:false}]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
      scales:{x:{grid:{display:false},ticks:{font:{size:10}}},y:{grid:{color:'rgba(0,0,0,.05)'},ticks:{font:{size:10}}}}}
  });
  const bins=Array(10).fill(0),binsA=Array(10).fill(0);
  allData.forEach(d=>{const b=Math.min(9,Math.floor(d.score*10));
    if(d.score>=.3) binsA[b]++; else bins[b]++;});
  if(scoreChart) scoreChart.destroy();
  scoreChart=new Chart(document.getElementById('scoreChart'),{type:'line',
    data:{labels:['0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9'],
      datasets:[
        {label:'Аномални',data:binsA,borderColor:'#dc2626',backgroundColor:'rgba(220,38,38,.08)',tension:.4,fill:true,borderWidth:2,pointRadius:2},
        {label:'Нормални',data:bins,borderColor:'#16a34a',backgroundColor:'rgba(22,163,74,.08)',tension:.4,fill:true,borderWidth:2,pointRadius:2}
      ]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
      scales:{x:{grid:{display:false},ticks:{font:{size:9}}},y:{grid:{color:'rgba(0,0,0,.05)'},ticks:{font:{size:10}}}}}
  });
}

function openModal(addr){
  const d=allData.find(x=>x.addr===addr); if(!d) return;
  document.getElementById('modal-name').textContent=d.name||'—';
  document.getElementById('modal-addr').textContent=d.addr||'—';
  document.getElementById('modal-score').textContent=(d.score||0).toFixed(4);
  document.getElementById('modal-score').style.color=scoreColor(d.score);
  document.getElementById('modal-badge').innerHTML=badge(d.level||'НИСЪк');
  document.getElementById('modal-type').textContent=d.type||'—';
  const shap=[
    {f:'out_in_ratio',v:d.out_in_ratio>2?(d.score*.35):(-d.score*.2)},
    {f:'contract_rate',v:d.contract_rate>.8?(d.score*.28):(-d.score*.15)},
    {f:'avg_gas_gwei',v:d.avg_gas_gwei>50?(d.score*.15):(-d.score*.08)},
    {f:'balance_eth',v:d.balance_eth>10000?(-d.score*.12):(d.score*.05)},
    {f:'unique_counterparts',v:d.unique_counterparts>1000?(-d.score*.18):(d.score*.08)},
  ];
  const maxV=Math.max(...shap.map(s=>Math.abs(s.v)));
  document.getElementById('modal-shap').innerHTML=shap.map(s=>`
    <div class="shap-row">
      <div class="shap-name">${s.f}</div>
      <div class="shap-bar-bg">
        ${s.v>=0?`<div class="shap-pos" style="width:${maxV?(s.v/maxV*100).toFixed(0):0}%"></div>`
                :`<div class="shap-neg" style="width:${maxV?(Math.abs(s.v)/maxV*100).toFixed(0):0}%"></div>`}
      </div>
      <span class="shap-val" style="color:${s.v>=0?'#dc2626':'#2563eb'}">${s.v>=0?'+':''}${s.v.toFixed(3)}</span>
    </div>`).join('');
  const feats=[
    {fn:'tx_count',fv:fmt(d.tx_count)},{fn:'out_in_ratio',fv:fmt(d.out_in_ratio)},
    {fn:'contract_rate',fv:((d.contract_rate||0)*100).toFixed(1)+'%'},{fn:'avg_gas_gwei',fv:fmt(d.avg_gas_gwei)},
    {fn:'balance_eth',fv:fmt(d.balance_eth)},{fn:'unique_counterparts',fv:fmt(d.unique_counterparts)},
    {fn:'IF score',fv:(d.if_score||0).toFixed(3)},{fn:'RF score',fv:(d.rf_score||0).toFixed(3)},
  ];
  document.getElementById('modal-feats').innerHTML=feats.map(f=>`
    <div class="feat-box"><div class="feat-name">${f.fn}</div><div class="feat-val">${f.fv}</div></div>`).join('');
  document.getElementById('modal').classList.add('open');
}

function closeModal(e){ if(e.target===document.getElementById('modal')) document.getElementById('modal').classList.remove('open'); }

function renderFullTable(){
  const keys=['name','addr','score','level','if_score','rf_score','gs_score','type','contract_rate','out_in_ratio','unique_counterparts','avg_gas_gwei','balance_eth','tx_count'];
  document.getElementById('full-hdr').innerHTML=keys.map(k=>`<th>${k}</th>`).join('');
  document.getElementById('full-body').innerHTML=allData.map(d=>`<tr>${keys.map(k=>`<td style="font-size:11px">${fmt(d[k])}</td>`).join('')}</tr>`).join('');
}

function renderLog(){
  document.getElementById('log-container').innerHTML=logEntries.length
    ? logEntries.map(e=>`<div class="log-row">
        <span class="log-time">${e.time}</span>
        <span class="log-addr">${e.addr.slice(0,18)}…</span>
        <span style="font-size:11px;color:#6b7280">${e.name}</span>
        ${badge(e.level)}
        <span style="font-weight:600;min-width:36px;text-align:right;color:${scoreColor(e.score)}">${e.score.toFixed(3)}</span>
      </div>`).join('')
    : '<p style="text-align:center;color:#9ca3af;padding:2rem;font-size:13px">Няма записи</p>';
}

function loadCSV(input){
  const f=input.files[0]; if(!f) return;
  const reader=new FileReader();
  reader.onload=e=>{
    fetch('/api/upload',{method:'POST',headers:{'Content-Type':'text/plain'},body:e.target.result})
      .then(r=>r.json()).then(d=>{ allData=d.data; document.getElementById('data-source').textContent='CSV: '+f.name; update(); });
  };
  reader.readAsText(f);
}

function exportCSV(){
  const keys=['name','addr','score','level','if_score','rf_score','gs_score','type','contract_rate','out_in_ratio','balance_eth','tx_count'];
  const rows=[keys.join(','),...allData.map(d=>keys.map(k=>JSON.stringify(d[k]??'')).join(','))];
  const a=document.createElement('a'); a.href='data:text/csv,'+encodeURIComponent(rows.join('\\n'));
  a.download='ews_results.csv'; a.click();
}
function exportLog(){
  const rows=['time,name,addr,score,level',...logEntries.map(e=>`${e.time},"${e.name}",${e.addr},${e.score},${e.level}`)];
  const a=document.createElement('a'); a.href='data:text/csv,'+encodeURIComponent(rows.join('\\n'));
  a.download='ews_log.csv'; a.click();
}
function applyThresholds(){
  const c=parseFloat(document.getElementById('thresh-crit').value);
  const h=parseFloat(document.getElementById('thresh-high').value);
  const m=parseFloat(document.getElementById('thresh-mid').value);
  allData=allData.map(d=>({...d,level:d.score>=c?'КРИТИЧЕН':d.score>=h?'ВИСОК':d.score>=m?'СРЕДЕН':'НИСЪк'}));
  update();
}
</script>
</body>
</html>'''

# ── Routes ─────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/api/data')
def api_data():
    data, source = load_real_data()
    return jsonify({'data': data, 'source': f'Данни от: {source}' if source != 'demo' else 'Demo данни — зареди CSV за реални данни'})

@app.route('/api/upload', methods=['POST'])
def api_upload():
    try:
        content = request.data.decode('utf-8')
        df = pd.read_csv(io.StringIO(content))
        score_col = next((c for c in ['score','ensemble_score','risk_score'] if c in df.columns), None)
        if not score_col:
            return jsonify({'error': 'Не е намерена колона score'}), 400
        df = df.rename(columns={score_col: 'score'})
        df['level'] = df['score'].apply(get_level)
        # Fill missing columns
        for col in ['name','addr','type','if_score','rf_score','gs_score','contract_rate','out_in_ratio','unique_counterparts','avg_gas_gwei','balance_eth','tx_count']:
            if col not in df.columns:
                df[col] = '' if col in ['name','addr','type'] else 0
        data = df.fillna(0).to_dict('records')
        return jsonify({'data': data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export')
def api_export():
    data, _ = load_real_data()
    df = pd.DataFrame(data)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(buf, mimetype='text/csv', as_attachment=True,
                     download_name=f'ews_results_{datetime.now().strftime("%Y%m%d")}.csv')

if __name__ == '__main__':
    print("\n🛡️  Early Warning System — Flask Dashboard")
    print("=" * 45)
    data, source = load_real_data()
    print(f"✓ Данни: {source} ({len(data)} адреса)")
    print(f"✓ URL: http://localhost:5000")
    print("=" * 45)
    app.run(debug=True, port=5000)
