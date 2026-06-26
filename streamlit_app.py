"""
Early Warning System — Streamlit Dashboard
==========================================
Стартиране:
    pip install streamlit pandas plotly shap scikit-learn
    streamlit run streamlit_app.py

Поставете файла в директорията на проекта (до src/ и data/).
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, json
from pathlib import Path

# ── Plotly за графиките ──────────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    st.warning("Инсталирай plotly: pip install plotly")

# ── Конфигурация ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Early Warning System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Стилове ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .stMetric { background: #f8f9fa; border-radius: 10px; padding: 0.5rem; }
    div[data-testid="metric-container"] { background: #f8f9fa; border: 1px solid #e9ecef;
        border-radius: 10px; padding: 0.75rem 1rem; }
    .risk-crit { color: #dc2626; font-weight: 600; }
    .risk-high { color: #d97706; font-weight: 600; }
    .risk-mid  { color: #2563eb; font-weight: 600; }
    .risk-low  { color: #16a34a; font-weight: 600; }
    .addr-mono { font-family: monospace; font-size: 0.8em; color: #6c757d; }
</style>
""", unsafe_allow_html=True)

# ── Помощни функции ───────────────────────────────────────────────────────
def get_level(score):
    if score >= 0.8: return "КРИТИЧЕН"
    if score >= 0.6: return "ВИСОК"
    if score >= 0.3: return "СРЕДЕН"
    return "НИСЪк"

def level_color(level):
    return {"КРИТИЧЕН": "#dc2626", "ВИСОК": "#d97706", "СРЕДЕН": "#2563eb", "НИСЪк": "#16a34a"}.get(level, "#6c757d")

def score_color(score):
    if score >= 0.8: return "#dc2626"
    if score >= 0.6: return "#d97706"
    if score >= 0.3: return "#2563eb"
    return "#16a34a"

def level_badge(level):
    colors = {"КРИТИЧЕН": ("#fef2f2","#dc2626"), "ВИСОК": ("#fffbeb","#d97706"),
              "СРЕДЕН": ("#eff6ff","#2563eb"), "НИСЪк": ("#f0fdf4","#16a34a")}
    bg, txt = colors.get(level, ("#f8f9fa","#6c757d"))
    return f'<span style="background:{bg};color:{txt};padding:2px 10px;border-radius:99px;font-size:12px;font-weight:500">{level}</span>'

# ── Demo данни ────────────────────────────────────────────────────────────
DEMO_DATA = [
    {"name":"Tornado Cash Router","addr":"0x905b63Fff465B9fFBF41DeA908CEb12478ec7A5","score":0.8409,"if_score":0.72,"rf_score":0.91,"gs_score":0.80,"type":"Mixer протокол","contract_rate":0.997,"out_in_ratio":4.21,"unique_counterparts":176,"avg_gas_gwei":42.3,"balance_eth":18590,"tx_count":12043},
    {"name":"Lazarus Group #1","addr":"0x3AD9dB589d201A710Ed237c829b7D0ae916e586","score":0.7821,"if_score":0.61,"rf_score":0.88,"gs_score":0.72,"type":"Санкциониран OFAC","contract_rate":0.672,"out_in_ratio":8.94,"unique_counterparts":34,"avg_gas_gwei":31.1,"balance_eth":0,"tx_count":289},
    {"name":"Rari Fuse Exploit","addr":"0x6EfF3310df9E73534c33F31A49a54cf2f45D2B0A","score":0.7234,"if_score":0.58,"rf_score":0.82,"gs_score":0.68,"type":"Exploit адрес","contract_rate":0.934,"out_in_ratio":7.12,"unique_counterparts":6,"avg_gas_gwei":189.3,"balance_eth":0,"tx_count":31},
    {"name":"Beanstalk Exploit","addr":"0x1c5dCdd006EA78a7E4783f9e6021C32935a10fb4","score":0.6914,"if_score":0.55,"rf_score":0.79,"gs_score":0.64,"type":"Exploit адрес","contract_rate":0.881,"out_in_ratio":6.33,"unique_counterparts":8,"avg_gas_gwei":220.4,"balance_eth":0,"tx_count":47},
    {"name":"Nomad Bridge","addr":"0x88A69B4E698A4B090DF6CF5Bd7B2D47325Ad30A3","score":0.5821,"if_score":0.41,"rf_score":0.65,"gs_score":0.54,"type":"Exploit адрес","contract_rate":0.912,"out_in_ratio":3.14,"unique_counterparts":44,"avg_gas_gwei":48.9,"balance_eth":0,"tx_count":1203},
    {"name":"Tornado Cash 100 ETH","addr":"0x910Cbd523D972eb0a6f4cAe4618aD62622b39DbF","score":0.1359,"if_score":0.22,"rf_score":0.11,"gs_score":0.14,"type":"Mixer протокол","contract_rate":1.0,"out_in_ratio":1.02,"unique_counterparts":12,"avg_gas_gwei":21.0,"balance_eth":0,"tx_count":4421},
    {"name":"Binance Exchange","addr":"0x28C6c06298d514Db089934071355E5743bf21d60","score":0.0206,"if_score":0.04,"rf_score":0.01,"gs_score":0.03,"type":"Борса (CEX)","contract_rate":0.123,"out_in_ratio":0.98,"unique_counterparts":9841,"avg_gas_gwei":18.2,"balance_eth":184200,"tx_count":2834199},
    {"name":"Kraken Exchange","addr":"0x2910543Af39abA0Cd09dBb2D50200b3E800A63D2","score":0.0618,"if_score":0.09,"rf_score":0.04,"gs_score":0.07,"type":"Борса (CEX)","contract_rate":0.284,"out_in_ratio":1.14,"unique_counterparts":212,"avg_gas_gwei":19.7,"balance_eth":662063,"tx_count":88441},
    {"name":"Aave Protocol","addr":"0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2","score":0.0431,"if_score":0.06,"rf_score":0.03,"gs_score":0.05,"type":"DeFi протокол","contract_rate":1.0,"out_in_ratio":0.89,"unique_counterparts":5621,"avg_gas_gwei":22.1,"balance_eth":1240000,"tx_count":4129831},
    {"name":"Compound Finance","addr":"0x3d9819210A31b4961b30EF54bE2aeD79B9c9Cd3B","score":0.0512,"if_score":0.07,"rf_score":0.04,"gs_score":0.06,"type":"DeFi протокол","contract_rate":1.0,"out_in_ratio":0.92,"unique_counterparts":3241,"avg_gas_gwei":20.8,"balance_eth":890000,"tx_count":2341000},
]

# ── Зареждане на реални данни ─────────────────────────────────────────────
def load_real_data():
    """Търси резултати от модела в стандартните пътища на проекта."""
    paths = [
        "results/ews/monitoring_log.csv",
        "results/ensemble_results.csv",
        "data/processed/real_features.csv",
        "../results/ews/monitoring_log.csv",
    ]
    for p in paths:
        if Path(p).exists():
            try:
                df = pd.read_csv(p)
                if 'score' in df.columns or 'ensemble_score' in df.columns or 'risk_score' in df.columns:
                    return df, p
            except: pass
    return None, None

# ── SIDEBAR ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ EWS Dashboard")
    st.markdown("---")
    page = st.radio("Навигация", ["📊 Dashboard", "📋 Адреси", "📁 Лог", "⚙️ Настройки"])
    st.markdown("---")
    st.markdown("**Данни**")

    use_demo = st.button("▶ Зареди Demo данни", use_container_width=True)
    uploaded = st.file_uploader("Зареди CSV", type=['csv'], label_visibility="collapsed")

    # Опит за автоматично зареждане на реални данни
    real_df, real_path = load_real_data()
    if real_path:
        st.success(f"✓ Намерен: {real_path}")

    st.markdown("---")
    st.markdown("**Прагове**")
    thresh_crit = st.slider("Критичен", 0.5, 1.0, 0.80, 0.05)
    thresh_high = st.slider("Висок", 0.3, 0.8, 0.60, 0.05)
    thresh_mid  = st.slider("Среден", 0.1, 0.5, 0.30, 0.05)

    st.markdown("---")
    st.caption("Crypto Anomaly Detection\nМагистърска теза 2026")

# ── Зареждане на данни ─────────────────────────────────────────────────────
@st.cache_data
def get_demo(): return pd.DataFrame(DEMO_DATA)

def custom_level(score, c, h, m):
    if score >= c: return "КРИТИЧЕН"
    if score >= h: return "ВИСОК"
    if score >= m: return "СРЕДЕН"
    return "НИСЪк"

if 'df' not in st.session_state:
    st.session_state.df = None

if use_demo:
    df = get_demo()
    df['level'] = df['score'].apply(lambda s: custom_level(s, thresh_crit, thresh_high, thresh_mid))
    st.session_state.df = df
elif uploaded:
    df = pd.read_csv(uploaded)
    score_col = next((c for c in ['score','ensemble_score','risk_score'] if c in df.columns), None)
    if score_col:
        df = df.rename(columns={score_col:'score'})
        df['level'] = df['score'].apply(lambda s: custom_level(s, thresh_crit, thresh_high, thresh_mid))
    st.session_state.df = df
elif real_df is not None and st.session_state.df is None:
    score_col = next((c for c in ['score','ensemble_score','risk_score'] if c in real_df.columns), None)
    if score_col:
        real_df = real_df.rename(columns={score_col:'score'})
        real_df['level'] = real_df['score'].apply(lambda s: custom_level(s, thresh_crit, thresh_high, thresh_mid))
    st.session_state.df = real_df

df = st.session_state.df

# ══════════════════════════════════════════════════════════════════════════
# DASHBOARD PAGE
# ══════════════════════════════════════════════════════════════════════════
if "Dashboard" in page:
    st.markdown("## Early Warning System")
    st.markdown("Crypto Anomaly Detection — Ethereum мрежа")

    if df is None:
        st.info("👆 Зареди CSV файл или натисни **Demo данни** от страничното меню.")
        st.stop()

    # Метрики
    total = len(df)
    score_col = 'score' if 'score' in df.columns else df.columns[0]
    anomal = len(df[df[score_col] >= thresh_mid])
    crit   = len(df[df[score_col] >= thresh_crit])
    normal = len(df[df[score_col] < thresh_mid])

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Анализирани адреси", total)
    c2.metric("Аномални (score ≥ " + str(thresh_mid) + ")", anomal, f"{anomal/total*100:.1f}%" if total else "—")
    c3.metric("Критични (score ≥ " + str(thresh_crit) + ")", crit, delta_color="inverse")
    c4.metric("Нормални", normal, f"{normal/total*100:.1f}%" if total else "—")

    st.divider()

    if HAS_PLOTLY:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Разпределение по риск нива")
            if 'level' in df.columns:
                lvl_counts = df['level'].value_counts().reindex(["НИСЪк","СРЕДЕН","ВИСОК","КРИТИЧЕН"], fill_value=0)
                fig = go.Figure(go.Bar(
                    x=lvl_counts.index, y=lvl_counts.values,
                    marker_color=["#16a34a","#2563eb","#d97706","#dc2626"],
                    text=lvl_counts.values, textposition='outside'
                ))
                fig.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=220,
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=False, xaxis=dict(showgrid=False), yaxis=dict(gridcolor='rgba(128,128,128,0.1)'))
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Score разпределение")
            if score_col in df.columns:
                fig2 = go.Figure()
                anom = df[df[score_col] >= thresh_mid][score_col]
                norm = df[df[score_col] < thresh_mid][score_col]
                if len(anom): fig2.add_trace(go.Histogram(x=anom, name='Аномални', marker_color='#dc2626', opacity=0.7, xbins=dict(size=0.1)))
                if len(norm): fig2.add_trace(go.Histogram(x=norm, name='Нормални', marker_color='#16a34a', opacity=0.7, xbins=dict(size=0.1)))
                fig2.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=220, barmode='overlay',
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    legend=dict(orientation='h',y=1.1))
                st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # Таблица
    st.markdown("#### Мониторирани адреси")
    c1,c2,c3 = st.columns([3,1,1])
    search = c1.text_input("Търси", placeholder="Адрес, тип...", label_visibility="collapsed")
    lv_filter = c2.selectbox("Ниво", ["Всички","КРИТИЧЕН","ВИСОК","СРЕДЕН","НИСЪк"], label_visibility="collapsed")
    sort_by = c3.selectbox("Сортирай", ["Score ↓","Score ↑"], label_visibility="collapsed")

    show_df = df.copy()
    if search:
        mask = show_df.apply(lambda r: search.lower() in str(r).lower(), axis=1)
        show_df = show_df[mask]
    if lv_filter != "Всички" and 'level' in show_df.columns:
        show_df = show_df[show_df['level'] == lv_filter]
    if score_col in show_df.columns:
        show_df = show_df.sort_values(score_col, ascending=(sort_by=="Score ↑"))

    st.caption(f"{len(show_df)} адреса")

    # Покажи топ колони
    disp_cols = [c for c in ['name','addr','score','level','if_score','rf_score','gs_score','type','contract_rate','out_in_ratio','balance_eth','tx_count'] if c in show_df.columns]
    styled = show_df[disp_cols].head(100)

    if 'score' in styled.columns:
        st.dataframe(
            styled,
            column_config={
                'score': st.column_config.ProgressColumn('Score', min_value=0, max_value=1, format="%.3f"),
                'if_score': st.column_config.NumberColumn('IF', format="%.2f"),
                'rf_score': st.column_config.NumberColumn('RF', format="%.2f"),
                'gs_score': st.column_config.NumberColumn('GS', format="%.2f"),
                'balance_eth': st.column_config.NumberColumn('Balance ETH', format="%,.0f"),
                'tx_count': st.column_config.NumberColumn('Tx Count', format="%,d"),
                'contract_rate': st.column_config.NumberColumn('Contract rate', format="%.2%"),
                'out_in_ratio': st.column_config.NumberColumn('Out/In ratio', format="%.2f"),
            },
            hide_index=True, use_container_width=True, height=400
        )
    else:
        st.dataframe(styled, hide_index=True, use_container_width=True)

    # Критични адреси
    if 'level' in df.columns:
        crit_df = df[df['level']=='КРИТИЧЕН']
        if len(crit_df) > 0:
            st.divider()
            st.markdown("#### 🚨 Критични адреси")
            for _, row in crit_df.iterrows():
                with st.expander(f"**{row.get('name','—')}** — score {row[score_col]:.4f}"):
                    c1,c2,c3 = st.columns(3)
                    c1.metric("Ensemble Score", f"{row[score_col]:.4f}")
                    if 'rf_score' in row: c2.metric("RF Score", f"{row['rf_score']:.3f}")
                    if 'contract_rate' in row: c3.metric("Contract Rate", f"{row['contract_rate']:.1%}")
                    if 'addr' in row:
                        st.code(row['addr'], language=None)
                    feat_cols = [c for c in ['out_in_ratio','unique_counterparts','avg_gas_gwei','balance_eth','tx_count','type'] if c in row.index]
                    if feat_cols:
                        st.dataframe(pd.DataFrame({'Характеристика':feat_cols,'Стойност':[row[c] for c in feat_cols]}), hide_index=True)

    # Експорт
    st.divider()
    if st.button("📥 Експорт CSV", use_container_width=False):
        csv = df.to_csv(index=False)
        st.download_button("⬇ Изтегли", csv, "ews_results.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════
# ADDRESSES PAGE
# ══════════════════════════════════════════════════════════════════════════
elif "Адреси" in page:
    st.markdown("## Всички адреси")
    if df is None:
        st.info("Зареди данни от страничното меню.")
    else:
        st.dataframe(df, hide_index=True, use_container_width=True, height=600)
        csv = df.to_csv(index=False)
        st.download_button("⬇ Изтегли CSV", csv, "all_addresses.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════
# LOG PAGE
# ══════════════════════════════════════════════════════════════════════════
elif "Лог" in page:
    st.markdown("## Мониторинг лог")
    log_paths = ["results/ews/monitoring_log.csv", "../results/ews/monitoring_log.csv"]
    log_df = None
    for p in log_paths:
        if Path(p).exists():
            log_df = pd.read_csv(p)
            break
    if log_df is not None:
        st.success(f"Намерен лог: {p}")
        st.dataframe(log_df.sort_values(log_df.columns[0], ascending=False) if len(log_df.columns) > 0 else log_df,
                     hide_index=True, use_container_width=True, height=500)
        csv = log_df.to_csv(index=False)
        st.download_button("⬇ Изтегли лог", csv, "monitoring_log.csv", "text/csv")
    elif df is not None and 'level' in df.columns:
        alerts = df[df['level'].isin(['КРИТИЧЕН','ВИСОК','СРЕДЕН'])].copy()
        st.dataframe(alerts.sort_values('score', ascending=False), hide_index=True, use_container_width=True)
    else:
        st.info("Файлът results/ews/monitoring_log.csv не е намерен. Стартирай 07_early_warning.py за да генерираш лог.")

# ══════════════════════════════════════════════════════════════════════════
# SETTINGS PAGE
# ══════════════════════════════════════════════════════════════════════════
elif "Настройки" in page:
    st.markdown("## Настройки")
    st.markdown("#### Прагове на риск скоринг")
    st.info(f"Текущи прагове: Критичен ≥ {thresh_crit} | Висок ≥ {thresh_high} | Среден ≥ {thresh_mid}")
    st.markdown("Промени ги от страничното меню.")
    st.markdown("#### Пътища за данни")
    for p in ["results/ews/monitoring_log.csv","results/ensemble_results.csv","data/processed/real_features.csv"]:
        exists = Path(p).exists()
        st.markdown(f"{'✅' if exists else '❌'} `{p}`")
    st.markdown("#### Формат на CSV файла")
    st.code("""
address,score,if_score,rf_score,gs_score,level,type,contract_rate,out_in_ratio,...
0x905b63...,0.8409,0.72,0.91,0.80,КРИТИЧЕН,Mixer протокол,0.997,4.21,...
    """)
