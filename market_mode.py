"""
Market Mode — AI-Powered Regime-Aware Stock Screener
Detects current market regime → identifies where money is flowing →
surfaces trade ideas that align with regime + flow simultaneously.

Free APIs only. No paid data required.
"""

import streamlit as st
st.set_page_config(
    page_title="Market Mode",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

import requests, json, time, re, threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from google import genai
from google.genai import types

# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* BASE */
*, *::before, *::after { box-sizing: border-box; font-style: normal !important; }
html, body, [data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section,
[data-testid="stAppViewContainer"] > section > div,
.main, .main > div {
    background-color: #f7f8fa !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: #111827 !important;
}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {
    background: #ffffff !important;
    border-right: 1px solid #e5e7eb !important;
}
[data-testid="stSidebar"] * { color: #374151 !important; }
[data-testid="stSidebar"] strong { color: #111827 !important; font-weight: 600 !important; }
.block-container { padding: 1.4rem 2.2rem 4rem !important; max-width: 1440px !important; }
em, i { font-style: normal !important; color: inherit !important; }
strong, b { font-weight: 600 !important; color: #111827 !important; }
a { color: #2563eb !important; }

/* TOPBAR */
.mm-top {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 0 16px; border-bottom: 1px solid #e5e7eb; margin-bottom: 20px;
}
.mm-logo { font-size: 1.2rem; font-weight: 800; letter-spacing: -0.8px; color: #111827 !important; }
.mm-logo .ac { color: #2563eb !important; }
.mm-meta { font-size: 0.7rem; color: #9ca3af !important; letter-spacing: 0.8px; text-transform: uppercase; }

/* REGIME BANNER */
.regime-banner {
    border-radius: 12px; padding: 20px 24px; margin: 0 0 22px;
    border: 1px solid; position: relative; overflow: hidden;
}
.regime-banner.risk_on       { background: #f0fdf4; border-color: #86efac; }
.regime-banner.risk_off      { background: #fff7ed; border-color: #fcd34d; }
.regime-banner.stagflation   { background: #fef2f2; border-color: #fca5a5; }
.regime-banner.rotation      { background: #eff6ff; border-color: #93c5fd; }
.regime-banner.crash         { background: #fff1f2; border-color: #fb7185; }
.regime-banner.unknown       { background: #f8fafc; border-color: #e5e7eb; }

.regime-name { font-size: 1.05rem; font-weight: 800; letter-spacing: -0.3px; margin-bottom: 4px; }
.regime-name.risk_on     { color: #15803d !important; }
.regime-name.risk_off    { color: #92400e !important; }
.regime-name.stagflation { color: #991b1b !important; }
.regime-name.rotation    { color: #1d4ed8 !important; }
.regime-name.crash       { color: #9f1239 !important; }
.regime-name.unknown     { color: #374151 !important; }

.regime-sub  { font-size: 0.82rem; color: #374151 !important; line-height: 1.5; }
.regime-conf { font-size: 0.72rem; font-weight: 700; color: #6b7280 !important; margin-top: 8px; }

/* FLOW BAR */
.flow-bar {
    display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 18px;
}
.flow-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 12px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; border: 1px solid;
    white-space: nowrap;
}
.flow-pill.in    { background: #f0fdf4; border-color: #86efac; color: #15803d !important; }
.flow-pill.out   { background: #fff1f2; border-color: #fca5a5; color: #9f1239 !important; }
.flow-pill.neut  { background: #f8fafc; border-color: #e5e7eb; color: #6b7280 !important; }

/* MACRO STRIP */
.macro-strip { display: flex; gap: 6px; margin-bottom: 18px; overflow-x: auto; padding-bottom: 2px; }
.macro-strip::-webkit-scrollbar { height: 3px; }
.macro-strip::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 2px; }
.mbox { flex: 1; min-width: 80px; background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 9px 10px; text-align: center; flex-shrink: 0; }
.mbox:hover { border-color: #2563eb; }
.mlbl   { font-size: 0.57rem; color: #9ca3af !important; text-transform: uppercase; letter-spacing: 1.2px; display: block; margin-bottom: 3px; font-weight: 600; }
.mprice { font-size: 0.9rem; font-weight: 700; color: #111827 !important; font-family: 'JetBrains Mono', monospace; display: block; }
.up  { color: #16a34a !important; font-size: 0.66rem; font-weight: 600; }
.dn  { color: #dc2626 !important; font-size: 0.66rem; font-weight: 600; }
.fl  { color: #9ca3af !important; font-size: 0.66rem; }

/* TRADE IDEA CARD */
.idea-card {
    background: #ffffff; border: 1px solid #e5e7eb;
    border-radius: 12px; padding: 18px 22px; margin: 8px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: box-shadow 0.12s, border-color 0.12s;
}
.idea-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-color: #2563eb; }
.idea-rank {
    display: inline-flex; align-items: center; justify-content: center;
    width: 28px; height: 28px; border-radius: 8px;
    font-size: 0.8rem; font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    flex-shrink: 0;
}
.idea-rank.s1 { background: #fef9c3; color: #713f12 !important; }
.idea-rank.s2 { background: #f0fdf4; color: #14532d !important; }
.idea-rank.s3 { background: #eff6ff; color: #1e3a8a !important; }
.idea-rank.sx { background: #f3f4f6; color: #374151 !important; }

.idea-header { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; flex-wrap: wrap; }
.idea-ticker { font-size: 1.15rem; font-weight: 800; color: #111827 !important; font-family: 'JetBrains Mono', monospace; }
.idea-score  { font-size: 0.72rem; font-weight: 700; padding: 2px 8px; border-radius: 5px; background: #f3f4f6; color: #374151 !important; font-family: 'JetBrains Mono', monospace; }
.idea-score.hi  { background: #dcfce7; color: #14532d !important; }
.idea-score.med { background: #fef9c3; color: #713f12 !important; }
.idea-fit    { font-size: 0.7rem; font-weight: 700; padding: 2px 8px; border-radius: 5px; letter-spacing: 0.5px; }
.idea-fit.strong { background: #eff6ff; color: #1d4ed8 !important; }
.idea-fit.good   { background: #f0fdf4; color: #15803d !important; }

.idea-price  { font-size: 1.0rem; font-weight: 700; color: #111827 !important; font-family: 'JetBrains Mono', monospace; }
.idea-chg    { font-size: 0.78rem; font-weight: 600; margin-left: 4px; }
.idea-chg.up { color: #16a34a !important; }
.idea-chg.dn { color: #dc2626 !important; }

.idea-signals { display: flex; gap: 6px; flex-wrap: wrap; margin: 8px 0; }
.sig-tag {
    font-size: 0.7rem; font-weight: 600; padding: 2px 8px;
    border-radius: 4px; background: #f3f4f6; color: #374151 !important;
    border: 1px solid #e5e7eb;
}
.sig-tag.bull { background: #f0fdf4; color: #15803d !important; border-color: #bbf7d0; }
.sig-tag.bear { background: #fff1f2; color: #9f1239 !important; border-color: #fecaca; }
.sig-tag.neut { background: #eff6ff; color: #1d4ed8 !important; border-color: #bfdbfe; }

.idea-levels {
    display: flex; gap: 18px; margin-top: 10px; flex-wrap: wrap;
    padding-top: 10px; border-top: 1px solid #f3f4f6;
}
.level-item { text-align: center; }
.level-lbl  { font-size: 0.6rem; color: #9ca3af !important; text-transform: uppercase; letter-spacing: 1px; display: block; font-weight: 600; margin-bottom: 2px; }
.level-val  { font-size: 0.88rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.level-val.entry  { color: #1d4ed8 !important; }
.level-val.target { color: #15803d !important; }
.level-val.stop   { color: #dc2626 !important; }
.level-val.upside { color: #15803d !important; }
.level-val.risk   { color: #dc2626 !important; }

.idea-thesis { font-size: 0.82rem; color: #374151 !important; line-height: 1.65; margin-top: 8px; }

/* SECTION LABEL */
.mm-label {
    font-size: 0.63rem; font-weight: 700; letter-spacing: 1.8px;
    text-transform: uppercase; color: #9ca3af !important;
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 6px; margin: 20px 0 14px; display: block;
}

/* SECTOR TABLE */
.sector-row {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 12px; background: #ffffff; border: 1px solid #e5e7eb;
    border-radius: 8px; margin: 4px 0;
}
.sector-name { font-size: 0.82rem; font-weight: 600; color: #111827 !important; min-width: 55px; }
.sector-bar-wrap { flex: 1; height: 6px; background: #f3f4f6; border-radius: 3px; overflow: hidden; }
.sector-bar { height: 100%; border-radius: 3px; }
.sector-pct { font-size: 0.78rem; font-weight: 700; min-width: 52px; text-align: right; font-family: 'JetBrains Mono', monospace; }
.sector-flow { font-size: 0.68rem; font-weight: 700; padding: 1px 6px; border-radius: 3px; min-width: 38px; text-align: center; }
.flow-in  { background: #dcfce7; color: #14532d !important; }
.flow-out { background: #fee2e2; color: #991b1b !important; }
.flow-neut { background: #f3f4f6; color: #6b7280 !important; }

/* INPUTS / BUTTONS */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #ffffff !important; border: 1.5px solid #d1d5db !important;
    border-radius: 8px !important; color: #111827 !important; font-size: 0.9rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #2563eb !important; box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: #ffffff !important; border: 1.5px solid #d1d5db !important; border-radius: 8px !important;
}
div.stButton > button { border-radius: 8px !important; font-weight: 600 !important; font-size: 0.85rem !important; }
div.stButton > button[kind="primary"] {
    background: #2563eb !important; border: 1px solid #2563eb !important; color: #ffffff !important;
}
div.stButton > button[kind="primary"]:hover { background: #1d4ed8 !important; }

/* MARKDOWN — kill italic, fix overflow */
div[data-testid="stMarkdownContainer"] * { font-style: normal !important; }
div[data-testid="stMarkdownContainer"] em { font-style: normal !important; color: #374151 !important; }
div[data-testid="stMarkdownContainer"] p  { color: #374151 !important; line-height: 1.75 !important; margin: 6px 0 !important; }
div[data-testid="stMarkdownContainer"] li { color: #374151 !important; line-height: 1.65 !important; }
div[data-testid="stMarkdownContainer"] strong { color: #111827 !important; font-weight: 700 !important; }
div[data-testid="stMarkdownContainer"] h2 { color: #111827 !important; font-weight: 700 !important; border-bottom: 1px solid #f3f4f6 !important; padding-bottom: 4px !important; }
div[data-testid="stMarkdownContainer"] h3 { color: #1e40af !important; font-weight: 700 !important; }
div[data-testid="stMarkdownContainer"] table { display: block !important; overflow-x: auto !important; border-collapse: collapse !important; width: 100% !important; }
div[data-testid="stMarkdownContainer"] th { background: #f3f4f6 !important; color: #111827 !important; padding: 8px 12px !important; border: 1px solid #e5e7eb !important; font-weight: 600 !important; }
div[data-testid="stMarkdownContainer"] td { padding: 7px 12px !important; border: 1px solid #e5e7eb !important; color: #374151 !important; }
div[data-testid="stMarkdownContainer"] tr:nth-child(even) td { background: #f9fafb !important; }
div[data-testid="stMarkdownContainer"] code { background: #f3f4f6 !important; color: #1f2937 !important; padding: 1px 5px !important; border-radius: 4px !important; font-size: 0.84em !important; }

/* TABS */
[data-testid="stTabs"] [role="tablist"] { border-bottom: 2px solid #e5e7eb !important; }
[data-testid="stTabs"] button { font-size: 0.8rem !important; font-weight: 500 !important; color: #6b7280 !important; padding: 8px 16px !important; border-bottom: 2px solid transparent !important; border-radius: 0 !important; background: transparent !important; margin-bottom: -2px !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #111827 !important; font-weight: 700 !important; border-bottom: 2px solid #2563eb !important; }
[data-testid="stMetricValue"] { color: #111827 !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #6b7280 !important; font-size: 0.8rem !important; }
hr { border: none !important; border-top: 1px solid #e5e7eb !important; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 3px; }

/* MOBILE */
@media (max-width: 768px) {
    .block-container { padding: 0.8rem 1rem 2.5rem !important; }
    .idea-card { padding: 14px 16px; }
    .regime-banner { padding: 14px 16px; }
    .idea-levels { gap: 12px; }
    .macro-strip { gap: 4px; }
    .mbox { min-width: 70px; }
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for k, v in [("regime_data", None), ("flow_data", None), ("ideas", []),
             ("last_run", None), ("sector_scores", {}), ("macro_snap", {})]:
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — config
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("**◎ Market Mode**")
    st.divider()

    gemini_key  = st.text_input("Gemini API Key", type="password", placeholder="AIza...", key="si_gemini",
                                help="Free · aistudio.google.com/app/apikey")
    tg_token    = st.text_input("Telegram Token", type="password", placeholder="optional", key="si_tg")
    tg_chat     = st.text_input("Telegram Chat ID", placeholder="optional", key="si_chat")

    st.divider()
    st.markdown("**Screening Universe**")
    use_sp500    = st.checkbox("S&P 500", value=True, key="cb_sp500")
    use_watchlist= st.checkbox("Custom watchlist", value=True, key="cb_wl")
    custom_raw   = st.text_area("Custom tickers", value="NVDA,TSM,PLTR,IONQ,RKLB,HIMS,NVO,MELI,AXTI,CELH",
                                height=70, key="ta_custom")
    custom_tickers = [s.strip().upper() for s in custom_raw.split(",") if s.strip()]

    st.divider()
    st.markdown("**Screener Settings**")
    min_score      = st.slider("Min signal score", 0, 10, 6, key="sl_minscore")
    max_ideas      = st.slider("Max ideas to show", 5, 30, 12, key="sl_maxideas")
    require_regime = st.checkbox("Regime-fit required", value=True, key="cb_regime")

    st.divider()
    if gemini_key:
        st.markdown('<span style="color:#16a34a;font-size:0.78rem">● AI ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#dc2626;font-size:0.78rem">● No API key</span>', unsafe_allow_html=True)
    if st.button("Clear all", use_container_width=True, key="btn_clear"):
        for k in ["regime_data","flow_data","ideas","last_run","sector_scores","macro_snap"]:
            st.session_state[k] = None if k != "ideas" else []
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LAYER
# ══════════════════════════════════════════════════════════════════════════════
YF = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
      "Accept": "application/json"}

@st.cache_data(ttl=300, show_spinner=False)
def yf_price(sym):
    try:
        r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&range=5d",
                         headers=YF, timeout=6)
        c = [x for x in r.json()["chart"]["result"][0]["indicators"]["quote"][0]["close"] if x]
        return (round(c[-1],2), round((c[-1]/c[-2]-1)*100,2)) if len(c)>=2 else (None,None)
    except: return None, None

@st.cache_data(ttl=300, show_spinner=False)
def yf_history(sym, rng="3mo"):
    try:
        r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&range={rng}",
                         headers=YF, timeout=8)
        d = r.json()["chart"]["result"][0]
        closes = [c for c in d["indicators"]["quote"][0]["close"] if c is not None]
        vols   = [v for v in d["indicators"]["quote"][0].get("volume",[]) if v is not None]
        return closes, vols, d.get("meta",{})
    except: return [], [], {}

@st.cache_data(ttl=600, show_spinner=False)
def yf_options_pc(sym):
    """Get put/call ratio from Yahoo options chain."""
    try:
        r = requests.get(f"https://query1.finance.yahoo.com/v7/finance/options/{sym}",
                         headers=YF, timeout=8)
        root = r.json().get("optionChain",{}).get("result",[])
        if not root: return None
        cv = pv = 0
        for block in root[:3]:
            for ob in block.get("options",[]):
                cv += sum(c.get("volume") or 0 for c in ob.get("calls",[]))
                pv += sum(c.get("volume") or 0 for c in ob.get("puts",[]))
        return round(pv/max(cv,1),3) if cv else None
    except: return None

@st.cache_data(ttl=300, show_spinner=False)
def get_macro():
    syms = {
        "SPX":   "%5EGSPC",
        "VIX":   "%5EVIX",
        "10Y":   "%5ETNX",
        "2Y":    "%5EIRX",
        "DXY":   "DX-Y.NYB",
        "Gold":  "GC%3DF",
        "Oil":   "CL%3DF",
        "HYG":   "HYG",   # high yield credit
        "IWM":   "IWM",   # small caps
    }
    out = {}
    for lbl, sym in syms.items():
        p, c = yf_price(sym)
        out[lbl] = (p, c)
    # BTC
    try:
        r = requests.get("https://api.coingecko.com/api/v3/simple/price",
            params={"ids":"bitcoin","vs_currencies":"usd","include_24hr_change":"true"},
            timeout=5, headers={"User-Agent":"MarketMode/1.0"})
        d = r.json()["bitcoin"]
        out["BTC"] = (round(d["usd"],0), round(d["usd_24h_change"],2))
    except: out["BTC"] = (None,None)
    return out

# S&P 500 tickers — 503 current members
SP500 = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK.B","TSLA","UNH",
    "XOM","LLY","JPM","JNJ","V","PG","MA","AVGO","HD","MRK","COST","CVX","ABBV",
    "MCD","PEP","KO","CRM","BAC","ACN","WMT","ADBE","TMO","CSCO","PFE","LIN",
    "NFLX","NKE","AMD","TXN","NEE","ORCL","DIS","RTX","BMY","AMGN","DHR","QCOM",
    "UNP","HON","INTC","PM","T","LOW","SBUX","IBM","CAT","GE","MDT","C","BA",
    "SPGI","COP","INTU","AMAT","ELV","DE","ISRG","AXP","GS","BLK","ADI","MU",
    "MDLZ","ADP","REGN","GILD","TJX","BKNG","SYK","MMM","ZTS","CVS","LMT","VRTX",
    "NOW","PANW","CI","LRCX","CB","ITW","BSX","SO","SLB","AON","CME","NSC",
    "ETN","PLD","WM","DUK","EMR","F","APD","HUM","ICE","CL","PNC","USB","FCX",
    "MCO","EW","KLAC","MSI","MO","TGT","HCA","NOC","SHW","ECL","FDX","PSA",
    "ORLY","CARR","AEP","CTAS","MMC","GD","ROP","SNPS","CDNS","FTNT","MCHP",
    "NXPI","STZ","HLT","AFL","A","PPG","IDXX","DLTR","YUM","PAYX","FAST",
    "KMB","VRSK","CSX","OTIS","WST","O","WBA","XEL","DOW","VICI","PWR","CTSH",
    "TRV","SPG","OXY","D","AIG","MTB","KEYS","GLW","LHX","HAL","IP","DVN",
    "FANG","MPC","VLO","EOG","HES","PSX","CVI","APA","EXC","ED","FE","AWK",
    "AMT","CCI","EQIX","DLR","WELL","VTR","EQR","AvB","MAA","UDR",
    "IRM","SBA","SBAC","PEG","ES","ETR","WEC","LNT","CNP","NLOK","TRMB",
    "GRMN","ZBRA","BALL","PKG","SEE","SON","WRK","MAS","LW","GPC","POOL",
    "ROL","SWK","IR","IEX","TT","XYL","GNRC","EXPD","CHRW","JBHT","ODFL",
    "CPRT","KNX","FRT","KIM","REG","CPT","NDAQ","CBOE","MKTX","PFG","RJF",
    "SIVB","ZION","FHN","CMA","RF","HBAN","WAL","FITB","KEY","CFG","STI",
    "NTRS","BK","STT","TROW","IVZ","AMG","LNC","UNM","PRU","MET","HIG","ALL",
    "WRB","CINF","GL","RGA","AIZ","RE","EG","ACGL","RLI","MMI","BRO","AJG",
    "FNF","FG","SFM","NWSA","NWS","FOXA","FOX","IPG","OMC","PVH","TAP","MOS",
    "CF","FMC","ALB","LYB","CE","EMN","PPL","NI","CMS","EVRG","AES","NRG",
    "PCG","SRE","MDU","OGE","PNW","WTRG","AWR","SJW","YORW",
    # Tech / growth adds
    "SNOW","DDOG","NET","CRWD","ZS","MDB","GTLB","ESTC","SPLK","OKTA",
    "TWLO","ZM","DOCN","HUBS","TTD","ROKU","PINS","LYFT","UBER","ABNB",
    "DASH","COIN","HOOD","SQ","PYPL","AFRM","SOFI","UPST","LC","OPEN",
    "RDFN","Z","CSGP","EXPI","COMP","FROG","PATH","AI","BBAI","SOUN",
    "IONQ","RGTI","QUBT","ARQQ","QMCO","MSTYV",
    # Defence / space
    "RKLB","ASTS","MNTS","SPCE","LMT","NOC","GD","RTX","HEI","TDG","CW",
    "KTOS","CACI","SAIC","BAH","LDOS","MANT","DRS","VEC","FLIR",
    # Healthcare / biotech
    "HIMS","TDOC","ACCD","ONEM","AMWL","OSCR","CLOV","HLTH",
    "MRNA","BNTX","NVAX","INO","OCGN","SRPT","BLUE","EDIT","CRSP","NTLA",
    "BEAM","PRIME","VERV","ARKG","XBI","IBB",
    # EV / clean energy
    "RIVN","LCID","NIO","LI","XPEV","NKLA","BLNK","CHPT","EVGO","VLTA",
    "STEM","ARRY","ENPH","SEDG","RUN","FSLR","NOVA","SHLS",
    # Commodities / materials
    "NEM","AEM","GOLD","KGC","AUY","WPM","PAAS","SVM","CDE","HL",
    "FCX","SCCO","HBM","TECK","BHP","RIO","VALE","CLF","X","NUE","STLD",
    "CCJ","LEU","UEC","UUUU","URG","NXE","EU",
    # Consumer
    "CELH","MNST","KDP","FIZZ","COKE","SAM","BUD","STZ","DKNG","PENN",
    "MGM","CZR","WYNN","LVS","VICI","RRR","CHDN",
]
SP500 = list(dict.fromkeys(SP500))  # deduplicate

SECTOR_ETFS = {
    "XLK":  "Technology",
    "XLF":  "Financials",
    "XLE":  "Energy",
    "XLV":  "Healthcare",
    "XLI":  "Industrials",
    "XLY":  "Cons. Discretionary",
    "XLP":  "Cons. Staples",
    "XLU":  "Utilities",
    "XLRE": "Real Estate",
    "XLB":  "Materials",
    "XLC":  "Comm. Services",
    "GDX":  "Gold Miners",
    "COPX": "Copper Miners",
    "ITB":  "Homebuilders",
    "XAR":  "Aerospace/Defense",
    "ARKK": "Innovation",
    "IBB":  "Biotech",
    "SMH":  "Semiconductors",
    "IWM":  "Small Caps",
    "TLT":  "Long Bonds",
    "GLD":  "Gold",
}

# ══════════════════════════════════════════════════════════════════════════════
#  QUANTITATIVE ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def calc_rsi(closes, n=14):
    if len(closes) < n+1: return 50.0
    a = np.array(closes, dtype=float); d = np.diff(a)
    g = np.where(d>0,d,0.0); l = np.where(d<0,-d,0.0)
    ag = np.mean(g[-n:]); al = np.mean(l[-n:])
    return round(float(100-100/(1+ag/(al+1e-9))),1)

def calc_macd(closes, f=12, s=26, sig=9):
    if len(closes)<s+sig: return None, None, None
    arr = pd.Series(closes, dtype=float)
    ema_f = arr.ewm(span=f,adjust=False).mean()
    ema_s = arr.ewm(span=s,adjust=False).mean()
    macd  = ema_f - ema_s
    signal= macd.ewm(span=sig,adjust=False).mean()
    hist  = macd - signal
    return round(float(macd.iloc[-1]),3), round(float(signal.iloc[-1]),3), round(float(hist.iloc[-1]),3)

def calc_adr(closes, n=14):
    """Average daily range as % — proxy for volatility."""
    if len(closes)<n+1: return None
    pct = [abs(closes[i]/closes[i-1]-1)*100 for i in range(1,len(closes))]
    return round(float(np.mean(pct[-n:])),2)

def calc_volume_ratio(vols, n=20):
    """Current vol vs n-day average."""
    if len(vols)<n+1 or vols[-1] is None: return None
    avg = np.mean(vols[-n-1:-1])
    return round(vols[-1]/max(avg,1),2) if avg > 0 else None

def calc_momentum(closes, periods=(5,10,20)):
    """Returns dict of momentum over given periods."""
    out = {}
    for p in periods:
        if len(closes) > p:
            out[p] = round((closes[-1]/closes[-p-1]-1)*100,2)
    return out

def score_stock(closes, vols, meta, rsi, ml, ms, mh, price, regime):
    """
    Regime-aware scoring. Returns (score 0-10, signals[], verdict, fit_to_regime).
    """
    score = 5.0
    sigs  = []
    w52h  = meta.get("fiftyTwoWeekHigh",0) or 0
    w52l  = meta.get("fiftyTwoWeekLow",0)  or 0
    mom   = calc_momentum(closes)
    vol_r = calc_volume_ratio(vols)

    # ── RSI ──────────────────────────────────────────────────────────────
    regime_rsi_bull = regime in ("risk_on","rotation")
    if rsi < 30:
        sigs.append(("RSI Oversold","bull")); score += 2.5
    elif rsi < 38:
        sigs.append(("RSI Low","bull")); score += 1.2
    elif rsi > 72:
        sigs.append(("RSI Overbought","bear")); score -= 2.0
    elif rsi > 62:
        sigs.append(("RSI Elevated","bear")); score -= 0.8

    # ── MACD ─────────────────────────────────────────────────────────────
    if ml is not None and ms is not None:
        if ml > ms and (mh or 0) > 0:
            sigs.append(("MACD Bull Cross","bull")); score += 1.2
        elif ml > ms:
            sigs.append(("MACD Bullish","bull")); score += 0.6
        elif ml < ms and (mh or 0) < 0:
            sigs.append(("MACD Bear Cross","bear")); score -= 1.2
        else:
            sigs.append(("MACD Bearish","bear")); score -= 0.5

    # ── Price vs MAs ──────────────────────────────────────────────────────
    if len(closes) >= 50:
        ma50 = float(np.mean(closes[-50:]))
        if price > ma50 * 1.02:
            sigs.append(("Above 50MA","bull")); score += 0.8
        elif price < ma50 * 0.97:
            sigs.append(("Below 50MA","bear")); score -= 0.8
    if len(closes) >= 200:
        ma200 = float(np.mean(closes[-200:]))
        if price > ma200:
            sigs.append(("Above 200MA","bull")); score += 0.5
        else:
            sigs.append(("Below 200MA","bear")); score -= 0.5

    # ── 52W position ──────────────────────────────────────────────────────
    if w52h > 0 and w52l > 0:
        pct = (price - w52l) / max(w52h - w52l, 0.01)
        if pct < 0.20:
            sigs.append(("Near 52W Low","bull")); score += 2.0
        elif pct < 0.35:
            sigs.append(("Lower Range","bull")); score += 1.0
        elif pct > 0.90:
            sigs.append(("Near 52W High","neut")); score += 0.5

    # ── Momentum ──────────────────────────────────────────────────────────
    m5  = mom.get(5,0)
    m20 = mom.get(20,0)
    if m5 > 5:   sigs.append((f"+{m5:.1f}% 1W","bull")); score += 0.7
    if m20 > 15: sigs.append((f"+{m20:.1f}% 1M","bull")); score += 0.8
    if m20 < -15: sigs.append((f"{m20:.1f}% 1M","bear")); score -= 0.8

    # ── Volume surge ──────────────────────────────────────────────────────
    if vol_r and vol_r > 2.5:
        sigs.append((f"Vol {vol_r:.1f}× avg","bull")); score += 1.2
    elif vol_r and vol_r > 1.5:
        sigs.append((f"Vol {vol_r:.1f}× avg","neut")); score += 0.5

    # ── Regime fit bonus/penalty ──────────────────────────────────────────
    regime_fit = "good"
    if regime == "risk_on":
        if rsi < 65 and (ml or 0) > (ms or 0) and price > float(np.mean(closes[-50:])) if len(closes)>=50 else True:
            score += 0.8; regime_fit = "strong"
        elif rsi > 70:
            score -= 0.5
    elif regime == "risk_off":
        if m20 and m20 < -5: score += 0.5; regime_fit = "strong"  # selling into safety
        if rsi < 35: score += 0.5
    elif regime == "stagflation":
        # Already filtered by sector
        regime_fit = "strong"
    elif regime == "rotation":
        if rsi < 40 and (vol_r or 0) > 1.5:
            score += 1.0; regime_fit = "strong"
        else:
            regime_fit = "good"

    score = round(max(0, min(10, score)), 1)
    verdict = ("STRONG BUY" if score >= 7.5 else "BUY" if score >= 6.5 else
               "WATCH" if score >= 5 else "CAUTION" if score >= 3.5 else "AVOID")
    return score, sigs, verdict, regime_fit

# ══════════════════════════════════════════════════════════════════════════════
#  REGIME ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def detect_regime(macro):
    """
    Classifies market regime from live macro data.
    Returns: (regime_key, confidence, evidence, description)
    """
    spx_p,  spx_c  = macro.get("SPX",  (None,None))
    vix_p,  vix_c  = macro.get("VIX",  (None,None))
    y10_p,  y10_c  = macro.get("10Y",  (None,None))
    y2_p,   y2_c   = macro.get("2Y",   (None,None))
    dxy_p,  dxy_c  = macro.get("DXY",  (None,None))
    hyg_p,  hyg_c  = macro.get("HYG",  (None,None))
    iwm_p,  iwm_c  = macro.get("IWM",  (None,None))
    gold_p, gold_c = macro.get("Gold", (None,None))

    # Get SPX 200MA
    spx_closes, _, spx_meta = yf_history("%5EGSPC", "1y")
    ma200 = float(np.mean(spx_closes[-200:])) if len(spx_closes)>=200 else None
    spx_above_200 = (spx_p > ma200) if (spx_p and ma200) else None

    # Yield curve
    spread_10_2 = round((y10_p or 0) - (y2_p or 0),3) if y10_p and y2_p else None
    inverted = spread_10_2 < -0.2 if spread_10_2 is not None else False

    evidence = []
    score = {"risk_on":0, "risk_off":0, "stagflation":0, "rotation":0, "crash":0}

    # ── VIX ───────────────────────────────────────────────────────────────
    if vix_p:
        if vix_p < 16:
            score["risk_on"] += 2; evidence.append(f"VIX {vix_p:.1f} — low fear")
        elif vix_p < 20:
            score["risk_on"] += 1; evidence.append(f"VIX {vix_p:.1f} — calm")
        elif vix_p < 26:
            score["rotation"] += 1; score["risk_off"] += 1; evidence.append(f"VIX {vix_p:.1f} — elevated")
        elif vix_p < 35:
            score["risk_off"] += 2; evidence.append(f"VIX {vix_p:.1f} — high fear")
        else:
            score["crash"] += 3; evidence.append(f"VIX {vix_p:.1f} — CRISIS LEVEL")

    # ── SPX vs 200MA ──────────────────────────────────────────────────────
    if spx_above_200 is True:
        score["risk_on"] += 2; evidence.append("SPX above 200MA — bull trend intact")
    elif spx_above_200 is False:
        score["risk_off"] += 2; score["crash"] += 1; evidence.append("SPX below 200MA — bear trend")

    # ── Yield curve ───────────────────────────────────────────────────────
    if spread_10_2 is not None:
        if inverted:
            score["risk_off"] += 1; score["stagflation"] += 1
            evidence.append(f"Yield curve inverted ({spread_10_2:+.2f}%) — recession signal")
        elif spread_10_2 > 0.5:
            score["risk_on"] += 1; score["rotation"] += 1
            evidence.append(f"Yield curve steepening ({spread_10_2:+.2f}%) — growth signal")

    # ── 10Y yield trend ───────────────────────────────────────────────────
    if y10_p and y10_c:
        if y10_p > 4.5 and (y10_c or 0) > 0:
            score["stagflation"] += 2; evidence.append(f"10Y yield {y10_p:.2f}% rising — inflation pressure")
        elif y10_p < 4.0 and (y10_c or 0) < 0:
            score["risk_off"] += 1; evidence.append(f"10Y yield falling — flight to safety")

    # ── DXY ───────────────────────────────────────────────────────────────
    if dxy_p and dxy_c:
        if (dxy_c or 0) > 0.5:
            score["stagflation"] += 1; score["risk_off"] += 1
            evidence.append(f"Dollar strengthening — risk-off rotation")
        elif (dxy_c or 0) < -0.5:
            score["risk_on"] += 1; evidence.append(f"Dollar weakening — risk appetite")

    # ── Credit spreads (HYG proxy) ────────────────────────────────────────
    if hyg_c:
        if (hyg_c or 0) < -1.0:
            score["crash"] += 2; score["risk_off"] += 1
            evidence.append(f"HYG falling {hyg_c:.1f}% — credit stress")
        elif (hyg_c or 0) > 0.5:
            score["risk_on"] += 1; evidence.append(f"HYG rising — credit healthy")

    # ── Small caps (IWM) ──────────────────────────────────────────────────
    if iwm_c and spx_c:
        if (iwm_c or 0) > (spx_c or 0) + 0.5:
            score["risk_on"] += 1; score["rotation"] += 1
            evidence.append("Small caps outperforming — risk appetite / rotation signal")
        elif (iwm_c or 0) < (spx_c or 0) - 0.5:
            score["risk_off"] += 1; evidence.append("Large caps outperforming — defensive positioning")

    # ── Gold ──────────────────────────────────────────────────────────────
    if gold_c:
        if (gold_c or 0) > 1.0:
            score["risk_off"] += 1; score["stagflation"] += 1
            evidence.append(f"Gold rising {gold_c:.1f}% — inflation/safety demand")

    # ── Classify ──────────────────────────────────────────────────────────
    best_regime = max(score, key=score.get)
    best_score  = score[best_regime]
    total       = sum(score.values()) or 1
    confidence  = round(min(95, best_score / total * 100 + 20), 0)

    regime_info = {
        "risk_on": {
            "name": "RISK-ON GROWTH",
            "desc": "Bull market regime — growth and momentum strategies outperform. "
                    "Money flowing into technology, discretionary, industrials, small caps.",
            "favours": ["XLK","XLY","XLI","SMH","ARKK","IWM"],
            "avoid":   ["XLU","XLP","TLT","GLD"],
            "screen":  "RSI 35–65, MACD bullish, above 50MA, momentum positive",
        },
        "risk_off": {
            "name": "RISK-OFF DEFENSIVE",
            "desc": "Defensive regime — capital preservation. Money rotating into "
                    "bonds, gold, utilities, staples. Avoid high-beta growth.",
            "favours": ["XLU","XLP","XLV","GLD","TLT","XLF"],
            "avoid":   ["ARKK","XLY","SMH","IWM"],
            "screen":  "Low beta, quality balance sheet, RSI < 55, below 52W high",
        },
        "stagflation": {
            "name": "STAGFLATION",
            "desc": "High inflation, slowing growth. Real assets and commodity producers "
                    "outperform. Bonds and rate-sensitive equities suffer.",
            "favours": ["XLE","XLB","GDX","COPX","GLD","OIL"],
            "avoid":   ["TLT","ARKK","XLY","XLRE"],
            "screen":  "Commodity exposure, pricing power, real asset backing",
        },
        "rotation": {
            "name": "SECTOR ROTATION",
            "desc": "Money moving between sectors. Look for laggard sectors showing early "
                    "accumulation signals. Divergence between sectors is high.",
            "favours": ["Sector bottom-fishers","Volume surges","RSI recovery"],
            "avoid":   ["Overbought leaders"],
            "screen":  "RSI 28–45, volume surge 1.5×+, sector bottom formation",
        },
        "crash": {
            "name": "CRISIS / CRASH",
            "desc": "Elevated systemic risk. Defensive and hedging strategies dominate. "
                    "Avoid new longs — focus on quality names and hedges.",
            "favours": ["GLD","TLT","VIXY","Cash"],
            "avoid":   ["High-beta growth","Leveraged ETFs"],
            "screen":  "Balance sheet quality, cash/debt > 1, revenue stability",
        },
        "unknown": {
            "name": "MIXED SIGNALS",
            "desc": "Market regime unclear — conflicting indicators. "
                    "Reduce position size, wait for clearer signal.",
            "favours": [],
            "avoid":   [],
            "screen":  "High quality only, small size",
        }
    }

    if best_score < 2:
        best_regime = "unknown"

    info = regime_info.get(best_regime, regime_info["unknown"])
    return best_regime, int(confidence), evidence[:6], info, score

# ══════════════════════════════════════════════════════════════════════════════
#  FLOW ENGINE — where is money actually moving
# ══════════════════════════════════════════════════════════════════════════════
def detect_flows():
    """
    Measures money flow across sectors using:
    1. 1W vs 1M performance delta (acceleration)
    2. Volume ratio vs average
    3. Options P/C ratio
    Returns sorted list of {etf, name, m1pct, w1pct, delta, vol_r, pc, flow_strength}
    """
    results = []
    for etf, name in SECTOR_ETFS.items():
        try:
            closes, vols, _ = yf_history(etf, "3mo")
            if len(closes) < 25: continue

            # Performance
            m1  = round((closes[-1]/closes[-22]-1)*100, 2) if len(closes)>=22 else 0
            w1  = round((closes[-1]/closes[-6]-1)*100,  2) if len(closes)>=6  else 0
            delta = round(w1 - m1/4, 2)  # acceleration vs trend

            # Volume
            vol_r = calc_volume_ratio(vols, 20)

            # Options P/C (use ETF)
            pc = yf_options_pc(etf)

            # Flow strength composite
            flow = 0.0
            flow += min(max(w1 * 0.4, -3), 3)
            flow += min(max(delta * 0.3, -2), 2)
            if vol_r and vol_r > 1.0: flow += min((vol_r-1)*0.5, 1.5)
            if pc:
                if pc < 0.6:   flow += 1.0   # heavy calls = bullish flow
                elif pc > 1.5: flow -= 1.0   # heavy puts = outflow
            flow = round(flow, 2)

            results.append({
                "etf": etf, "name": name,
                "m1": m1, "w1": w1, "delta": delta,
                "vol_r": vol_r, "pc": pc,
                "flow": flow,
            })
            time.sleep(0.08)
        except: continue

    results.sort(key=lambda x: x["flow"], reverse=True)
    return results

# ══════════════════════════════════════════════════════════════════════════════
#  SCREENER — regime-aware stock scoring
# ══════════════════════════════════════════════════════════════════════════════
def build_universe(use_sp500, custom_tickers):
    """Build the screening universe."""
    universe = []
    if use_sp500:
        universe.extend(SP500)
    if custom_tickers:
        universe.extend(custom_tickers)
    # Always include sector ETFs
    universe.extend(list(SECTOR_ETFS.keys()))
    return list(dict.fromkeys(universe))  # deduplicate, preserve order

def screen_universe(universe, regime, regime_info, flow_data, min_score,
                    require_regime_fit, max_results, progress_cb=None):
    """
    Screen the universe against regime + flow signals.
    Returns sorted list of trade candidates.
    """
    # Determine regime-favoured sectors
    favoured_etfs  = set(regime_info.get("favours", []))
    favoured_names = {SECTOR_ETFS.get(e,"") for e in favoured_etfs}

    # Top flowing sectors by name
    top_flow_names = {r["name"] for r in flow_data[:5]} if flow_data else set()

    results = []
    total   = len(universe)

    for idx, sym in enumerate(universe):
        if progress_cb: progress_cb(idx, total, sym)
        try:
            closes, vols, meta = yf_history(sym, "6mo")
            if len(closes) < 25: continue

            price, chg = yf_price(sym)
            if not price: continue

            rsi = calc_rsi(closes)
            ml, ms, mh = calc_macd(closes)
            mom = calc_momentum(closes)

            score, sigs, verdict, fit = score_stock(
                closes, vols, meta, rsi, ml, ms, mh, price, regime)

            if score < min_score: continue
            if require_regime_fit and verdict == "AVOID": continue

            w52h = meta.get("fiftyTwoWeekHigh",0) or 0
            w52l = meta.get("fiftyTwoWeekLow",0)  or 0

            # Entry / target / stop calculation
            adr = calc_adr(closes) or 1.5
            vol_r = calc_volume_ratio(vols) or 1.0

            # Dynamic levels based on technicals
            entry  = round(price * 0.995, 2)  # slight pullback entry
            stop   = round(max(price * 0.88, price - adr * 3), 2)
            # Target: based on regime and score
            tgt_mult = 1.15 + (score - 5) * 0.03
            if regime == "risk_on": tgt_mult += 0.05
            target = round(price * tgt_mult, 2)
            upside = round((target/price - 1)*100, 1)
            downside = round((stop/price - 1)*100, 1)
            rr = round(upside / abs(downside), 2) if downside < 0 else 0

            results.append({
                "ticker":  sym,
                "price":   price,
                "chg":     chg or 0,
                "score":   score,
                "verdict": verdict,
                "fit":     fit,
                "rsi":     rsi,
                "macd":    ml,
                "signals": sigs,
                "entry":   entry,
                "target":  target,
                "stop":    stop,
                "upside":  upside,
                "downside":downside,
                "rr":      rr,
                "m1":      mom.get(20,0),
                "w1":      mom.get(5,0),
                "vol_r":   round(vol_r,2),
                "w52h":    w52h,
                "w52l":    w52l,
            })
            time.sleep(0.05)
        except: continue

    results.sort(key=lambda x: (x["score"], x["rr"]), reverse=True)
    return results[:max_results]

# ══════════════════════════════════════════════════════════════════════════════
#  AI SYNTHESIS — Gemini trade thesis generator
# ══════════════════════════════════════════════════════════════════════════════
def call_gemini(system, user, key):
    if not key: return "[No API key]"
    try:
        client = genai.Client(api_key=key)
        srch   = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(
            tools=[srch], temperature=0.3, system_instruction=system)
        return client.models.generate_content(
            model="gemini-2.5-flash", contents=user, config=config).text
    except Exception as e: return f"[API error: {e}]"

def stream_gemini(system, user, key):
    if not key: yield "⚠️ No API key."; return
    try:
        client = genai.Client(api_key=key)
        resp = client.models.generate_content_stream(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(system_instruction=system, temperature=0.3),
            contents=user)
        for chunk in resp:
            if chunk.text: yield chunk.text
    except Exception as e: yield f"\n❌ {e}"

def generate_regime_thesis(regime_key, regime_info, evidence, macro, flow_data, top_ideas, key):
    """Generate streaming AI commentary on regime + top trade ideas."""
    macro_str = "\n".join([f"{k}: {v[0]} ({'+' if (v[1] or 0)>=0 else ''}{v[1]:.2f}%)"
                           for k,v in macro.items() if v[0]])
    flow_str  = "\n".join([f"{r['etf']} ({r['name']}): 1M={r['m1']:+.1f}% 1W={r['w1']:+.1f}% flow={r['flow']:+.1f}"
                           for r in flow_data[:8]])
    ideas_str = "\n".join([f"{i['ticker']}: score={i['score']} RSI={i['rsi']} entry=${i['entry']} target=${i['target']} stop=${i['stop']}"
                           for i in top_ideas[:6]])
    evidence_str = "\n".join(evidence)

    return stream_gemini(
        "You are the Chief Market Strategist at a tier-1 hedge fund. "
        "Synthesize regime data, sector flows, and quantitative signals into precise, "
        "actionable trade theses. Be specific — exact entries, targets, stops, catalysts. "
        "No generic commentary. Every sentence should contain a number or a specific name.",
        f"""Date: {datetime.now():%Y-%m-%d}

DETECTED REGIME: {regime_info['name']}
CONFIDENCE: {st.session_state.get('regime_confidence',50)}%
REGIME EVIDENCE:
{evidence_str}

LIVE MACRO DATA:
{macro_str}

SECTOR FLOW SCORES (positive = inflow, negative = outflow):
{flow_str}

TOP QUANTITATIVE SCREENER HITS:
{ideas_str}

REGIME CONTEXT: {regime_info['desc']}
FAVOURED SECTORS: {', '.join(regime_info.get('favours',[]))}

Provide:

## Market Regime Assessment
2-3 sentences. What is the market telling us RIGHT NOW? What is the dominant force?

## Where Money Is Flowing
Which 2-3 sectors show the clearest institutional accumulation? Use the flow data above.
Specific ETF + reasoning why this flow is real and not just noise.

## Top 3 Trade Theses
For each of the top screener hits, write a complete trade thesis:

**[TICKER]** — [one-line thesis]
- Why this aligns with current regime
- Specific catalyst (use live web search for upcoming events)
- Entry: $X | Target: $X | Stop: $X | Horizon: X weeks/months
- Risk: what kills this trade

## The Contrarian Check
Which of the above ideas is most likely to fail and why?
What macro event would flip the regime and invalidate these trades?

## One Number That Matters Most This Week
[specific indicator, level, and what it means for these trades]""",
        key
    )

def generate_single_thesis(idea, regime_key, regime_info, key):
    return stream_gemini(
        "You are a senior analyst writing a precise, data-driven investment thesis. "
        "Every claim must be backed by a number or a named catalyst. No filler.",
        f"""Date: {datetime.now():%Y-%m-%d}
TICKER: {idea['ticker']}
CURRENT PRICE: ${idea['price']} ({idea['chg']:+.2f}% today)
REGIME: {regime_info['name']}

QUANTITATIVE DATA:
RSI: {idea['rsi']} | MACD: {idea['macd']} | Score: {idea['score']}/10
1M Performance: {idea['m1']:+.1f}% | 1W: {idea['w1']:+.1f}%
Volume ratio: {idea['vol_r']}×
52W High: ${idea['w52h']} | 52W Low: ${idea['w52l']}
Signals: {', '.join(s[0] for s in idea['signals'])}

PROPOSED LEVELS:
Entry: ${idea['entry']} | Target: ${idea['target']} | Stop: ${idea['stop']}
Upside: {idea['upside']:+.1f}% | Downside: {idea['downside']:.1f}% | R/R: {idea['rr']}×

Use live web search to find: recent news, upcoming earnings/catalysts, analyst targets.

Write a complete investment thesis:

## Business Snapshot
What does this company do? Who are its 3 biggest customers/revenue drivers?

## Why Now — Regime Alignment
Specifically how does {idea['ticker']} benefit from the current {regime_info['name']} regime?
What tailwind is in place RIGHT NOW?

## The Quantitative Setup
Interpret the technicals above. What does RSI {idea['rsi']} + this MACD signal + volume tell you?
Is the setup early-stage or mature?

## Upcoming Catalysts
What specific events (earnings date, FDA decision, contract announcement, macro event)
could be the price catalyst in the next 30-90 days?

## Supply Chain Alpha
Who are the Tier-1/2 suppliers that benefit if {idea['ticker']} wins?
Is there a smaller, underfollowed name that offers better asymmetry?

## Bear Case
What is the #1 risk? What price level invalidates this thesis?

## Final Trade Setup
Entry: ${idea['entry']} | Target: ${idea['target']} | Stop: ${idea['stop']}
Position size: X% of portfolio | Hold period: X months
Conviction: HIGH/MEDIUM based on regime alignment""",
        key
    )

def send_telegram(text, token, chat_id):
    if not token or not chat_id: return
    for chunk in [text[i:i+4000] for i in range(0,len(text),4000)]:
        try:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                         json={"chat_id":chat_id,"text":chunk}, timeout=10)
        except: pass

# ══════════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def render_idea_card(idea, rank, regime_key):
    rank_classes = {1:"s1",2:"s2",3:"s3"}
    rk = rank_classes.get(rank, "sx")
    sc_cls = "hi" if idea["score"]>=7.5 else "med" if idea["score"]>=6 else ""
    fit_cls = "strong" if idea["fit"]=="strong" else "good"
    chg_cls = "up" if idea["chg"]>=0 else "dn"
    sgn     = "+" if idea["chg"]>=0 else ""

    # Signal tags HTML
    tags = ""
    for sig_label, sig_type in idea["signals"][:5]:
        tags += f'<span class="sig-tag {sig_type}">{sig_label}</span>'

    st.markdown(f"""
<div class="idea-card">
  <div class="idea-header">
    <span class="idea-rank {rk}">#{rank}</span>
    <span class="idea-ticker">{idea['ticker']}</span>
    <span class="idea-score {sc_cls}">{idea['score']}/10</span>
    <span class="idea-fit {fit_cls}">{'✦ ' if idea['fit']=='strong' else ''}REGIME FIT</span>
    <span class="idea-price">${idea['price']:,.2f}</span>
    <span class="idea-chg {chg_cls}">{sgn}{idea['chg']:.2f}%</span>
  </div>
  <div class="idea-signals">{tags}</div>
  <div class="idea-levels">
    <div class="level-item"><span class="level-lbl">Entry</span><span class="level-val entry">${idea['entry']:,.2f}</span></div>
    <div class="level-item"><span class="level-lbl">Target</span><span class="level-val target">${idea['target']:,.2f}</span></div>
    <div class="level-item"><span class="level-lbl">Stop</span><span class="level-val stop">${idea['stop']:,.2f}</span></div>
    <div class="level-item"><span class="level-lbl">Upside</span><span class="level-val upside">{idea['upside']:+.1f}%</span></div>
    <div class="level-item"><span class="level-lbl">Downside</span><span class="level-val risk">{idea['downside']:.1f}%</span></div>
    <div class="level-item"><span class="level-lbl">R/R</span><span class="level-val">{idea['rr']}×</span></div>
  </div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN UI
# ══════════════════════════════════════════════════════════════════════════════

# ── TOP BAR ────────────────────────────────────────────────────────────────────
last_run = st.session_state.get("last_run")
last_str = last_run[:16].replace("T"," ") if last_run else "Not run yet"
st.markdown(f"""
<div class="mm-top">
  <div class="mm-logo">◎ Market<span class="ac">Mode</span></div>
  <div class="mm-meta">Regime-aware stock screener &nbsp;·&nbsp; {datetime.now().strftime("%d %b %Y · %H:%M UTC")} &nbsp;·&nbsp; Last run: {last_str}</div>
</div>""", unsafe_allow_html=True)

# ── MACRO STRIP ────────────────────────────────────────────────────────────────
with st.spinner(""):
    macro = get_macro()
    st.session_state["macro_snap"] = macro

strip = ""
for lbl, (p,c) in macro.items():
    if p:
        cls = "up" if (c or 0)>=0 else "dn"
        sgn = "+" if (c or 0)>=0 else ""
        fmt = f"{p:,.0f}" if p>500 else f"{p:.2f}"
        strip += f'<div class="mbox"><span class="mlbl">{lbl}</span><span class="mprice">{fmt}</span><span class="{cls}">{sgn}{c:.2f}%</span></div>'
    else:
        strip += f'<div class="mbox"><span class="mlbl">{lbl}</span><span class="mprice">—</span><span class="fl">—</span></div>'
st.markdown(f'<div class="macro-strip">{strip}</div>', unsafe_allow_html=True)

# ── MAIN SCAN BUTTON + CONTROLS ────────────────────────────────────────────────
ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2,1,1,2])
with ctrl1:
    run_btn = st.button("◎  Run Market Mode Scan", type="primary", key="btn_run", use_container_width=True)
with ctrl2:
    mode_override = st.selectbox("Override regime", ["Auto-detect","Risk-On","Risk-Off","Stagflation","Rotation","Crash"], key="sb_regime_override")
with ctrl3:
    show_ai = st.checkbox("AI synthesis", value=True, key="cb_ai")
with ctrl4:
    if st.session_state.get("ideas"):
        n = len(st.session_state["ideas"])
        r = st.session_state.get("regime_data",{}).get("name","—")
        st.markdown(f'<div style="padding:8px 12px;background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;font-size:0.82rem;color:#15803d;font-weight:600">✓ {n} ideas · {r}</div>', unsafe_allow_html=True)

# ── RUN SCAN ──────────────────────────────────────────────────────────────────
if run_btn:
    if not gemini_key:
        st.warning("Add your Gemini API key in the sidebar to run the scan.")
        st.stop()

    # 1. Detect regime
    with st.status("Detecting market regime...", expanded=False) as s1:
        regime_key, confidence, evidence, regime_info, raw_scores = detect_regime(macro)

        # Override if requested
        override_map = {"Risk-On":"risk_on","Risk-Off":"risk_off","Stagflation":"stagflation",
                        "Rotation":"rotation","Crash":"crash"}
        if mode_override != "Auto-detect":
            regime_key = override_map.get(mode_override, regime_key)
            confidence = 100

        st.session_state["regime_data"] = regime_info
        st.session_state["regime_key"]  = regime_key
        st.session_state["regime_conf"] = confidence
        st.session_state["regime_evidence"] = evidence
        s1.update(label=f"✓ Regime: {regime_info['name']} ({confidence}% confidence)", state="complete")

    # 2. Detect flows
    with st.status("Scanning sector flows...", expanded=False) as s2:
        flow_data = detect_flows()
        st.session_state["flow_data"] = flow_data
        top3_flow = ", ".join(r["name"] for r in flow_data[:3])
        s2.update(label=f"✓ Top inflows: {top3_flow}", state="complete")

    # 3. Screen universe
    universe = build_universe(use_sp500, custom_tickers if use_watchlist else [])
    progress_bar = st.progress(0, f"Screening {len(universe)} stocks...")

    screened = []
    lock = threading.Lock()
    chunk_size = 50  # process in parallel chunks
    chunks = [universe[i:i+chunk_size] for i in range(0,len(universe),chunk_size)]

    processed = 0
    for chunk in chunks:
        def screen_chunk(c):
            return screen_universe(
                c, regime_key, regime_info, flow_data,
                min_score, require_regime, max_ideas*2, None)
        results = screen_chunk(chunk)
        with lock: screened.extend(results)
        processed += len(chunk)
        pct = min(int(processed/len(universe)*100), 100)
        progress_bar.progress(pct, f"Screened {processed}/{len(universe)} stocks...")

    # Sort and cap
    screened.sort(key=lambda x: (x["score"], x["rr"]), reverse=True)
    screened = screened[:max_ideas]
    st.session_state["ideas"] = screened
    st.session_state["last_run"] = datetime.now().isoformat()
    progress_bar.empty()

    # 4. Push to Telegram
    if tg_token and tg_chat and screened:
        msg = f"◎ MARKET MODE\n{datetime.now():%Y-%m-%d %H:%M}\n\n"
        msg += f"REGIME: {regime_info['name']} ({confidence}%)\n\n"
        msg += "TOP IDEAS:\n"
        for i, idea in enumerate(screened[:5],1):
            msg += f"#{i} {idea['ticker']} ${idea['price']} — Score {idea['score']}/10\n"
            msg += f"   Entry ${idea['entry']} | Target ${idea['target']} | Stop ${idea['stop']}\n"
        send_telegram(msg, tg_token, tg_chat)

    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS DISPLAY
# ══════════════════════════════════════════════════════════════════════════════
regime_key  = st.session_state.get("regime_key","unknown")
regime_info = st.session_state.get("regime_data") or {"name":"—","desc":"Run a scan to detect the current market regime.","favours":[],"avoid":[]}
confidence  = st.session_state.get("regime_conf",0)
evidence    = st.session_state.get("regime_evidence",[])
flow_data   = st.session_state.get("flow_data",[])
ideas       = st.session_state.get("ideas",[])

# ── REGIME BANNER ─────────────────────────────────────────────────────────────
conf_str = f" · {confidence}% confidence" if confidence else ""
evidence_html = " &nbsp;·&nbsp; ".join(evidence[:4]) if evidence else "Run a scan to detect regime"
st.markdown(f"""
<div class="regime-banner {regime_key}">
  <div class="regime-name {regime_key}">{regime_info['name']}{conf_str}</div>
  <div class="regime-sub">{regime_info['desc']}</div>
  <div class="regime-conf">{evidence_html}</div>
</div>""", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────
tab_ideas, tab_flows, tab_ai, tab_deep, tab_screen = st.tabs([
    f"◎ Trade Ideas ({len(ideas)})",
    "↗ Sector Flows",
    "⬡ AI Synthesis",
    "⊕ Deep Dive",
    "⚙ Screener Table",
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — TRADE IDEAS
# ══════════════════════════════════════════════════════════════════════════════
with tab_ideas:
    if not ideas:
        st.info("Run a scan to surface regime-aligned trade ideas.")
    else:
        # Flow pills
        if flow_data:
            st.markdown('<span class="mm-label">Money flowing into</span>', unsafe_allow_html=True)
            pills = ""
            for r in flow_data[:8]:
                cls = "in" if r["flow"] > 0.5 else "out" if r["flow"] < -0.5 else "neut"
                arr = "↑" if r["flow"] > 0.5 else "↓" if r["flow"] < -0.5 else "→"
                pills += f'<span class="flow-pill {cls}">{arr} {r["name"]} ({r["w1"]:+.1f}%)</span>'
            st.markdown(f'<div class="flow-bar">{pills}</div>', unsafe_allow_html=True)

        st.markdown(f'<span class="mm-label">Top {len(ideas)} regime-aligned trade ideas · sorted by score × risk/reward</span>', unsafe_allow_html=True)

        # Two-column card grid
        col_a, col_b = st.columns(2)
        for i, idea in enumerate(ideas):
            col = col_a if i%2==0 else col_b
            with col:
                render_idea_card(idea, i+1, regime_key)
                if st.button("⊕ Deep Dive", key=f"dd_{idea['ticker']}_{i}", use_container_width=False):
                    st.session_state["deep_ticker"] = idea
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — SECTOR FLOWS
# ══════════════════════════════════════════════════════════════════════════════
with tab_flows:
    if not flow_data:
        st.info("Run a scan to see sector flow data.")
    else:
        st.markdown('<span class="mm-label">Sector momentum & flow scores</span>', unsafe_allow_html=True)
        st.caption("Flow score = weighted composite of 1W acceleration, volume ratio, and options put/call. Positive = inflow, negative = outflow.")

        max_flow = max(abs(r["flow"]) for r in flow_data) or 1
        for r in flow_data:
            flow_cls  = "flow-in" if r["flow"] > 0.5 else "flow-out" if r["flow"] < -0.5 else "flow-neut"
            flow_lbl  = "INFLOW" if r["flow"] > 0.5 else "OUTFLOW" if r["flow"] < -0.5 else "NEUTRAL"
            bar_pct   = abs(r["flow"]) / max_flow * 100
            bar_color = "#16a34a" if r["flow"] > 0.5 else "#dc2626" if r["flow"] < -0.5 else "#9ca3af"
            m1_str    = f"{r['m1']:+.1f}%"
            w1_str    = f"{r['w1']:+.1f}%"
            pc_str    = f"P/C {r['pc']:.2f}" if r.get("pc") else "P/C —"

            st.markdown(f"""
<div class="sector-row">
  <span class="sector-name">{r['etf']}</span>
  <span style="font-size:0.75rem;color:#6b7280;min-width:130px">{r['name']}</span>
  <div class="sector-bar-wrap">
    <div class="sector-bar" style="width:{bar_pct:.0f}%;background:{bar_color}"></div>
  </div>
  <span class="sector-pct" style="color:{'#16a34a' if r['m1']>0 else '#dc2626'}">{m1_str}</span>
  <span class="sector-pct" style="color:{'#16a34a' if r['w1']>0 else '#dc2626'};font-size:0.75rem">{w1_str} 1W</span>
  <span style="font-size:0.7rem;color:#9ca3af;min-width:55px">{pc_str}</span>
  <span class="sector-flow {flow_cls}">{flow_lbl}</span>
</div>""", unsafe_allow_html=True)

        # Regime vs flow alignment check
        if regime_key != "unknown" and regime_info.get("favours"):
            st.markdown('<span class="mm-label">Regime vs flow alignment</span>', unsafe_allow_html=True)
            favoured = set(regime_info.get("favours",[]))
            flow_map = {r["etf"]: r["flow"] for r in flow_data}
            matches = [(etf, flow_map.get(etf,0)) for etf in favoured if etf in flow_map and flow_map[etf]>0]
            misses  = [(etf, flow_map.get(etf,0)) for etf in favoured if etf in flow_map and flow_map[etf]<=0]
            if matches:
                st.markdown("**✅ Regime-confirmed flows** (regime says BUY, flow confirms):")
                for etf,f in sorted(matches,key=lambda x:-x[1]):
                    st.markdown(f"- **{etf}** ({SECTOR_ETFS.get(etf,'')}) — flow score {f:+.2f}")
            if misses:
                st.markdown("**⚠️ Regime-divergent flows** (regime says BUY, but flow not yet confirming):")
                for etf,f in misses:
                    st.markdown(f"- {etf} ({SECTOR_ETFS.get(etf,'')}) — flow score {f:+.2f}")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — AI SYNTHESIS (streaming)
# ══════════════════════════════════════════════════════════════════════════════
with tab_ai:
    if not ideas:
        st.info("Run a scan first, then come back here for AI synthesis.")
    else:
        st.markdown("**AI reads the regime + flows + screener output and writes the full market narrative.**")
        if st.button("⬡ Generate AI Market Synthesis", type="primary", key="btn_ai_synth"):
            if not gemini_key:
                st.warning("Add Gemini API key.")
            else:
                with st.status("Streaming market synthesis...", expanded=True) as s:
                    out = st.empty(); full = ""
                    for chunk in generate_regime_thesis(
                        regime_key, regime_info, evidence, macro, flow_data, ideas, gemini_key):
                        full += chunk; out.markdown(full)
                    s.update(label="✓ Synthesis complete", state="complete")

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("⬇ Download", data=full,
                        file_name=f"market_mode_{datetime.now():%Y%m%d_%H%M}.txt",
                        mime="text/plain", use_container_width=True)
                with col2:
                    if tg_token and tg_chat:
                        if st.button("→ Telegram", key="btn_tg_ai", use_container_width=True):
                            send_telegram(f"◎ MARKET MODE AI\n{datetime.now():%Y-%m-%d %H:%M}\n\n{full}", tg_token, tg_chat)
                            st.success("Sent")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — DEEP DIVE (streaming per-stock thesis)
# ══════════════════════════════════════════════════════════════════════════════
with tab_deep:
    # Check if coming from card click
    deep_idea = st.session_state.get("deep_ticker")

    c1, c2 = st.columns([3,1])
    with c1:
        ticker_input = st.text_input(
            "Ticker for deep dive",
            value=deep_idea["ticker"] if deep_idea else "",
            placeholder="NVDA · RKLB · PLTR · any ticker...",
            key="ti_deep_ticker",
        )
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        deep_btn = st.button("⊕ Run Deep Dive", type="primary", key="btn_deep")

    if deep_btn and ticker_input.strip():
        if not gemini_key:
            st.warning("Add Gemini API key.")
        else:
            t = ticker_input.strip().upper().replace("$","")
            # Build the idea dict from screener or fresh data
            if deep_idea and deep_idea["ticker"] == t:
                idea = deep_idea
            else:
                # Build on-the-fly
                closes, vols, meta = yf_history(t, "6mo")
                price, chg = yf_price(t)
                if closes and price:
                    rsi = calc_rsi(closes)
                    ml,ms,mh = calc_macd(closes)
                    score, sigs, verdict, fit = score_stock(
                        closes, vols, meta, rsi, ml, ms, mh, price,
                        regime_key or "risk_on")
                    adr = calc_adr(closes) or 1.5
                    mom = calc_momentum(closes)
                    idea = {
                        "ticker":t,"price":price,"chg":chg or 0,"score":score,
                        "verdict":verdict,"fit":fit,"rsi":rsi,"macd":ml,"signals":sigs,
                        "entry":round(price*0.995,2),"target":round(price*1.18,2),
                        "stop":round(price*0.9,2),"upside":18.0,"downside":-10.0,"rr":1.8,
                        "m1":mom.get(20,0),"w1":mom.get(5,0),
                        "vol_r":calc_volume_ratio(vols) or 1.0,
                        "w52h":meta.get("fiftyTwoWeekHigh",0) or 0,
                        "w52l":meta.get("fiftyTwoWeekLow",0)  or 0,
                    }
                else:
                    st.error(f"Could not fetch data for {t}")
                    st.stop()

            # Render quick card
            render_idea_card(idea, 1, regime_key or "risk_on")

            # Stream thesis
            with st.status(f"Writing deep dive for {t}...", expanded=True) as ds:
                out = st.empty(); full = ""
                for chunk in generate_single_thesis(
                    idea, regime_key or "risk_on", regime_info, gemini_key):
                    full += chunk; out.markdown(full)
                ds.update(label=f"✓ Deep dive complete — {t}", state="complete")

            col1, col2 = st.columns(2)
            with col1:
                st.download_button("⬇ Download", data=full,
                    file_name=f"deepdive_{t}_{datetime.now():%Y%m%d_%H%M}.txt",
                    mime="text/plain", use_container_width=True)
            with col2:
                if tg_token and tg_chat:
                    if st.button("→ Telegram", key=f"tg_deep_{t}", use_container_width=True):
                        send_telegram(f"◎ DEEP DIVE: {t}\n{datetime.now():%Y-%m-%d %H:%M}\n\n{full}", tg_token, tg_chat)
                        st.success("Sent")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — FULL SCREENER TABLE
# ══════════════════════════════════════════════════════════════════════════════
with tab_screen:
    if not ideas:
        st.info("Run a scan to see the full screener output.")
    else:
        st.markdown(f'<span class="mm-label">Full screener results — {len(ideas)} ideas · regime: {regime_info["name"]}</span>', unsafe_allow_html=True)

        df = pd.DataFrame([{
            "Ticker":  i["ticker"],
            "Price":   i["price"],
            "1D%":     round(i["chg"],2),
            "1M%":     round(i["m1"],1),
            "RSI":     i["rsi"],
            "Score":   i["score"],
            "Verdict": i["verdict"],
            "Fit":     i["fit"].upper(),
            "Entry":   i["entry"],
            "Target":  i["target"],
            "Stop":    i["stop"],
            "Upside%": i["upside"],
            "R/R":     i["rr"],
            "Vol×":    i["vol_r"],
        } for i in ideas])

        def c_score(v):
            if isinstance(v,(int,float)):
                if v>=7.5: return "color:#14532d;font-weight:700;background:#dcfce7"
                if v>=6:   return "color:#713f12;font-weight:700;background:#fef9c3"
            return ""
        def c_rsi(v):
            if isinstance(v,(int,float)):
                if v<35: return "color:#14532d;font-weight:600"
                if v>70: return "color:#991b1b;font-weight:600"
            return ""
        def c_chg(v):
            if isinstance(v,(int,float)):
                return "color:#16a34a" if v>0 else "color:#dc2626" if v<0 else ""
            return ""
        def c_fit(v):
            if v=="STRONG": return "color:#1d4ed8;font-weight:700"
            return "color:#15803d"

        st.dataframe(
            df.style
              .map(c_score, subset=["Score"])
              .map(c_rsi,   subset=["RSI"])
              .map(c_chg,   subset=["1D%","1M%","Upside%"])
              .map(c_fit,   subset=["Fit"]),
            width="stretch", hide_index=True, height=500
        )

        st.download_button(
            "⬇ Download screener CSV",
            data=df.to_csv(index=False),
            file_name=f"market_mode_screener_{datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv",
            key="btn_dl_csv"
        )
