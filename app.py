"""
╔══════════════════════════════════════════════════════════════╗
║         PAIRS TRADING ENGINE  ·  Statistical Arbitrage       ║
║         OLS · ADF Cointegration · EWMA Z-Score               ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Pairs Trading Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
}
.stApp { background-color: #0a0e1a; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1120 0%, #0a0e1a 100%);
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Space Mono', monospace;
    color: #38bdf8;
}

/* ── Metric Cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0f1729 0%, #131d35 100%);
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 16px;
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover { border-color: #38bdf8; }
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.75rem !important; letter-spacing: 0.08em; text-transform: uppercase; }
[data-testid="stMetricValue"] { font-family: 'Space Mono', monospace; color: #f8fafc !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1120;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1e2d4a;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #64748b;
    background: transparent;
    border-radius: 7px;
    padding: 8px 18px;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e3a5f, #0f2a4a) !important;
    color: #38bdf8 !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    background: linear-gradient(135deg, #0369a1, #0284c7);
    color: #f0f9ff;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    transition: all 0.2s;
    letter-spacing: 0.05em;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0284c7, #0ea5e9);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(56,189,248,0.3);
}

/* ── Selectbox / Inputs ── */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: #0d1120 !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Sliders ── */
.stSlider > div > div > div > div { background: #38bdf8 !important; }

/* ── Signal Banners ── */
.signal-buy {
    background: linear-gradient(135deg, #052e16 0%, #064e3b 100%);
    border: 1px solid #10b981;
    border-left: 4px solid #10b981;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 8px 0;
}
.signal-sell {
    background: linear-gradient(135deg, #2d0a0a 0%, #450a0a 100%);
    border: 1px solid #ef4444;
    border-left: 4px solid #ef4444;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 8px 0;
}
.signal-neutral {
    background: linear-gradient(135deg, #0c1424 0%, #0f1e38 100%);
    border: 1px solid #3b5282;
    border-left: 4px solid #64748b;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 8px 0;
}
.signal-text { font-family: 'Space Mono', monospace; font-size: 1.05rem; font-weight: 700; }
.signal-sub { font-size: 0.82rem; color: #94a3b8; margin-top: 4px; }

/* ── Section Headers ── */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #38bdf8;
    border-bottom: 1px solid #1e2d4a;
    padding-bottom: 6px;
    margin: 20px 0 12px 0;
}

/* ── Stat Table ── */
.stat-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid #1a2540;
    font-size: 0.85rem;
}
.stat-label { color: #64748b; }
.stat-value { font-family: 'Space Mono', monospace; color: #e2e8f0; }
.stat-value.positive { color: #10b981; }
.stat-value.negative { color: #ef4444; }
.stat-value.warning  { color: #f59e0b; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1e2d4a; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  MATH ENGINE
# ══════════════════════════════════════════════════════════════════

class PairsMathEngine:
    """Core statistical engine: OLS hedge ratio, ADF, EWMA z-score."""

    @staticmethod
    def compute_hedge_ratio(y: pd.Series, x: pd.Series) -> tuple[float, float, object]:
        """OLS regression: y ~ β·x + α"""
        X = add_constant(x)
        model = OLS(y, X).fit()
        beta  = model.params.iloc[1]
        alpha = model.params.iloc[0]
        return beta, alpha, model

    @staticmethod
    def compute_spread(y: pd.Series, x: pd.Series, beta: float, alpha: float) -> pd.Series:
        return y - (beta * x + alpha)

    @staticmethod
    def adf_test(spread: pd.Series) -> dict:
        result = adfuller(spread.dropna(), autolag='AIC')
        return {
            "adf_stat":    result[0],
            "p_value":     result[1],
            "lags_used":   result[2],
            "n_obs":       result[3],
            "critical_1":  result[4]["1%"],
            "critical_5":  result[4]["5%"],
            "critical_10": result[4]["10%"],
            "stationary":  result[1] < 0.05,
        }

    @staticmethod
    def engle_granger_coint(y: pd.Series, x: pd.Series) -> dict:
        t_stat, p_value, crit = coint(y, x)
        return {"t_stat": t_stat, "p_value": p_value,
                "critical_1": crit[0], "critical_5": crit[1],
                "cointegrated": p_value < 0.05}

    @staticmethod
    def ewma_zscore(spread: pd.Series, halflife: int) -> pd.Series:
        mu    = spread.ewm(halflife=halflife).mean()
        sigma = spread.ewm(halflife=halflife).std()
        return (spread - mu) / sigma

    @staticmethod
    def rolling_zscore(spread: pd.Series, window: int) -> pd.Series:
        mu    = spread.rolling(window).mean()
        sigma = spread.rolling(window).std()
        return (spread - mu) / sigma

    @staticmethod
    def generate_signals(zscore: pd.Series, entry: float, exit_z: float) -> pd.Series:
        signals = pd.Series(0, index=zscore.index)
        signals[zscore >  entry] = -1   # Spread too wide → short
        signals[zscore < -entry] =  1   # Spread too narrow → long
        signals[(zscore > -exit_z) & (zscore < exit_z)] = 0
        return signals

    @staticmethod
    def compute_returns(spread: pd.Series, signals: pd.Series) -> pd.Series:
        spread_ret   = spread.diff()
        strat_ret    = signals.shift(1) * spread_ret
        return strat_ret

    @staticmethod
    def performance_metrics(returns: pd.Series) -> dict:
        returns = returns.dropna()
        if len(returns) == 0:
            return {}
        total_return  = returns.sum()
        ann_return    = returns.mean() * 252
        ann_vol       = returns.std()  * np.sqrt(252)
        sharpe        = ann_return / ann_vol if ann_vol != 0 else 0
        cum           = returns.cumsum()
        roll_max      = cum.cummax()
        drawdown      = cum - roll_max
        max_dd        = drawdown.min()
        n_trades      = (signals_from_returns(returns) != 0).sum()
        win_rate      = (returns[returns != 0] > 0).mean() if (returns != 0).any() else 0
        return {
            "total_return": total_return,
            "ann_return":   ann_return,
            "ann_vol":      ann_vol,
            "sharpe":       sharpe,
            "max_drawdown": max_dd,
            "win_rate":     win_rate,
        }


def signals_from_returns(ret: pd.Series) -> pd.Series:
    return (ret != 0).astype(int)


# ══════════════════════════════════════════════════════════════════
#  DATA LAYER  (synthetic / real yfinance)
# ══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def generate_synthetic_pair(
    n: int = 756,
    beta: float = 1.1,
    alpha: float = 5.0,
    noise_std: float = 2.5,
    drift: float = 0.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, str]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    # Asset X: geometric random walk
    log_ret_x = rng.normal(0.0003, 0.015, n)
    x = 100 * np.exp(np.cumsum(log_ret_x))
    # Asset Y: cointegrated with X + mean-reverting noise
    noise = rng.normal(0, noise_std, n)
    ou_noise = np.zeros(n)
    theta = 0.15
    for i in range(1, n):
        ou_noise[i] = ou_noise[i-1] + theta * (0 - ou_noise[i-1]) + noise[i]
    y = alpha + beta * x + ou_noise + drift * np.arange(n)
    df = pd.DataFrame({"Asset_X": x, "Asset_Y": y}, index=dates)
    return df, "Synthetic Cointegrated Pair"


@st.cache_data(show_spinner=False)
def fetch_real_data(ticker_x: str, ticker_y: str, period: str = "3y") -> tuple[pd.DataFrame | None, str]:
    try:
        import yfinance as yf
        raw = yf.download([ticker_x, ticker_y], period=period, auto_adjust=True, progress=False)["Close"]
        raw.columns = ["Asset_X", "Asset_Y"]
        raw = raw.dropna()
        return raw, f"{ticker_x} / {ticker_y}"
    except Exception as e:
        return None, str(e)


# ══════════════════════════════════════════════════════════════════
#  CHART HELPERS
# ══════════════════════════════════════════════════════════════════

COLORS = {
    "x":       "#38bdf8",
    "y":       "#a78bfa",
    "spread":  "#34d399",
    "zscore":  "#fb923c",
    "signal_long":  "#10b981",
    "signal_short": "#ef4444",
    "neutral": "#64748b",
    "grid":    "#1e2d4a",
    "bg":      "rgba(10,14,26,0)",
}

LAYOUT_BASE = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["bg"],
    font=dict(family="DM Sans", color="#94a3b8", size=11),
    margin=dict(l=10, r=10, t=36, b=10),
    legend=dict(bgcolor="rgba(13,17,32,0.8)", bordercolor=COLORS["grid"],
                borderwidth=1, font=dict(size=10)),
    xaxis=dict(gridcolor=COLORS["grid"], zeroline=False, showgrid=True),
    yaxis=dict(gridcolor=COLORS["grid"], zeroline=False, showgrid=True),
)


def chart_prices(df: pd.DataFrame, label: str) -> go.Figure:
    y_norm = df["Asset_Y"] / df["Asset_Y"].iloc[0] * 100
    x_norm = df["Asset_X"] / df["Asset_X"].iloc[0] * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=x_norm, name="Asset X (norm.)",
                             line=dict(color=COLORS["x"], width=1.8)))
    fig.add_trace(go.Scatter(x=df.index, y=y_norm, name="Asset Y (norm.)",
                             line=dict(color=COLORS["y"], width=1.8)))
    fig.update_layout(**LAYOUT_BASE, title=dict(text=f"📈 Normalised Price — {label}",
                      font=dict(family="Space Mono", size=13, color="#e2e8f0")))
    return fig


def chart_spread(spread: pd.Series, beta: float) -> go.Figure:
    mu = spread.mean()
    s1 = spread.std()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spread.index, y=spread, name="Spread",
                             line=dict(color=COLORS["spread"], width=1.5)))
    for mult, label, dash in [(0,"Mean","solid"),(1,"±1σ","dot"),(2,"±2σ","dash")]:
        for sign in ([0] if mult == 0 else [1,-1]):
            v = mu + sign * mult * s1
            fig.add_hline(y=v, line=dict(color="#64748b", dash=dash, width=1),
                          annotation=dict(text=label if sign >= 0 else "",
                                          font_color="#64748b", font_size=10, x=0))
    fig.update_layout(**LAYOUT_BASE, title=dict(text=f"📉 Spread  (β = {beta:.4f})",
                      font=dict(family="Space Mono", size=13, color="#e2e8f0")))
    return fig


def chart_zscore(zscore: pd.Series, entry: float, signals: pd.Series) -> go.Figure:
    fig = go.Figure()
    # background bands
    for y0, y1, col in [(-entry, entry, "rgba(56,189,248,0.04)"),
                         (entry, zscore.max()*1.1, "rgba(239,68,68,0.06)"),
                         (zscore.min()*1.1, -entry, "rgba(16,185,129,0.06)")]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=col, line_width=0)

    fig.add_trace(go.Scatter(x=zscore.index, y=zscore, name="EWMA Z-Score",
                             line=dict(color=COLORS["zscore"], width=1.8)))
    for v, col, lbl in [(entry, COLORS["signal_short"], f"+{entry}σ"),
                         (-entry, COLORS["signal_long"],  f"-{entry}σ"),
                         (0, "#475569", "0")]:
        fig.add_hline(y=v, line=dict(color=col, dash="dash", width=1),
                      annotation=dict(text=lbl, font_color=col, font_size=10, x=1))

    # trade markers
    long_idx  = signals[signals ==  1].index
    short_idx = signals[signals == -1].index
    if len(long_idx):
        fig.add_trace(go.Scatter(x=long_idx, y=zscore[long_idx], mode="markers",
                                 marker=dict(color=COLORS["signal_long"], size=7, symbol="triangle-up"),
                                 name="Long Signal"))
    if len(short_idx):
        fig.add_trace(go.Scatter(x=short_idx, y=zscore[short_idx], mode="markers",
                                 marker=dict(color=COLORS["signal_short"], size=7, symbol="triangle-down"),
                                 name="Short Signal"))
    fig.update_layout(**LAYOUT_BASE, title=dict(text="⚡ EWMA Z-Score & Signals",
                      font=dict(family="Space Mono", size=13, color="#e2e8f0")))
    return fig


def chart_equity(returns: pd.Series) -> go.Figure:
    cum = returns.cumsum()
    dd  = cum - cum.cummax()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32],
                        vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=cum.index, y=cum, name="Cumulative P&L",
                             line=dict(color=COLORS["spread"], width=2),
                             fill="tozeroy", fillcolor="rgba(52,211,153,0.08)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd, name="Drawdown",
                             line=dict(color="#ef4444", width=1.2),
                             fill="tozeroy", fillcolor="rgba(239,68,68,0.12)"), row=2, col=1)
    fig.update_layout(**LAYOUT_BASE, title=dict(text="💰 Equity Curve & Drawdown",
                      font=dict(family="Space Mono", size=13, color="#e2e8f0")),
                      showlegend=True)
    fig.update_xaxes(gridcolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"])
    return fig


def chart_regression(y: pd.Series, x: pd.Series, beta: float, alpha: float) -> go.Figure:
    x_line  = np.linspace(x.min(), x.max(), 200)
    y_line  = alpha + beta * x_line
    residuals = y - (alpha + beta * x)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers",
                             marker=dict(color=residuals, colorscale="RdYlGn",
                                         size=4, opacity=0.6,
                                         colorbar=dict(title="Residual")),
                             name="Observations"))
    fig.add_trace(go.Scatter(x=x_line, y=y_line, name=f"OLS: β={beta:.4f}",
                             line=dict(color="#38bdf8", width=2)))
    fig.update_layout(**LAYOUT_BASE, title=dict(text="📐 OLS Regression",
                      font=dict(family="Space Mono", size=13, color="#e2e8f0")),
                      xaxis_title="Asset X", yaxis_title="Asset Y")
    return fig


def chart_rolling_correlation(df: pd.DataFrame, window: int = 60) -> go.Figure:
    corr = df["Asset_X"].rolling(window).corr(df["Asset_Y"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=corr.index, y=corr, name=f"{window}-day Rolling Corr",
                             line=dict(color="#a78bfa", width=1.8),
                             fill="tozeroy", fillcolor="rgba(167,139,250,0.08)"))
    fig.add_hline(y=0.8, line=dict(color="#10b981", dash="dot", width=1),
                  annotation=dict(text="Strong (0.80)", font_color="#10b981", font_size=10))
    fig.add_hline(y=0.6, line=dict(color="#f59e0b", dash="dot", width=1),
                  annotation=dict(text="Moderate (0.60)", font_color="#f59e0b", font_size=10))
    fig.update_layout(**LAYOUT_BASE, title=dict(text=f"🔗 Rolling {window}-Day Correlation",
                      font=dict(family="Space Mono", size=13, color="#e2e8f0")))
    return fig


def chart_zscore_dist(zscore: pd.Series) -> go.Figure:
    clean = zscore.dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=clean, nbinsx=60, name="Z-Score",
                               marker_color="#fb923c", opacity=0.7,
                               histnorm="probability density"))
    mu, sigma = clean.mean(), clean.std()
    x_range = np.linspace(mu - 4*sigma, mu + 4*sigma, 300)
    fig.add_trace(go.Scatter(x=x_range,
                             y=stats.norm.pdf(x_range, mu, sigma),
                             name="N(μ,σ)", line=dict(color="#38bdf8", width=2)))
    fig.update_layout(**LAYOUT_BASE, title=dict(text="📊 Z-Score Distribution",
                      font=dict(family="Space Mono", size=13, color="#e2e8f0")),
                      bargap=0.05)
    return fig


# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
      <div style='font-family: Space Mono, monospace; font-size: 1.15rem; 
                  color: #38bdf8; letter-spacing: 0.08em;'>⚡ PAIRS ENGINE</div>
      <div style='font-size: 0.72rem; color: #475569; letter-spacing: 0.12em; 
                  margin-top: 4px;'>STATISTICAL ARBITRAGE v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">DATA SOURCE</div>', unsafe_allow_html=True)
    data_mode = st.radio("", ["🧪 Synthetic (Demo)", "📡 Live Market Data"],
                         label_visibility="collapsed")

    if "Synthetic" in data_mode:
        st.markdown('<div class="section-header">SIMULATION</div>', unsafe_allow_html=True)
        n_days     = st.slider("Trading Days", 252, 1500, 756, 63)
        true_beta  = st.slider("True β (Hedge Ratio)", 0.5, 2.5, 1.1, 0.05)
        true_alpha = st.slider("True α (Intercept)", -20.0, 20.0, 5.0, 0.5)
        noise_std  = st.slider("OU Noise σ", 0.5, 8.0, 2.5, 0.25)
        drift_val  = st.slider("Structural Drift", -0.02, 0.02, 0.0, 0.001,
                               help="Non-zero breaks cointegration over time")
        sim_seed   = st.number_input("Random Seed", 0, 9999, 42, 1)
    else:
        st.markdown('<div class="section-header">TICKERS</div>', unsafe_allow_html=True)
        st.caption("Requires `yfinance` installed. Examples: KO/PEP · GS/MS · XOM/CVX")
        ticker_x = st.text_input("Asset X Ticker", "KO").upper().strip()
        ticker_y = st.text_input("Asset Y Ticker", "PEP").upper().strip()
        period   = st.selectbox("Lookback Period", ["1y","2y","3y","5y"], index=2)
        n_days = true_beta = true_alpha = noise_std = drift_val = sim_seed = None

    st.markdown('<div class="section-header">STATISTICAL PARAMS</div>', unsafe_allow_html=True)
    halflife    = st.slider("EWMA Half-Life (days)", 5, 120, 30, 5,
                            help="Controls decay speed of EWMA weights")
    roll_window = st.slider("Rolling Window (days)", 20, 250, 60, 5,
                            help="Used for rolling z-score & correlation")
    entry_z     = st.slider("Entry Z-Score (σ)", 1.0, 4.0, 2.0, 0.1,
                            help="Signal threshold — higher = fewer, cleaner trades")
    exit_z      = st.slider("Exit Z-Score (σ)",  0.0, 2.0, 0.5, 0.1,
                            help="Position closed when |z| drops below this")
    adf_sig     = st.selectbox("ADF Significance", ["1%","5%","10%"], index=1)

    st.markdown('<div class="section-header">PORTFOLIO</div>', unsafe_allow_html=True)
    capital     = st.number_input("Capital ($)", 10_000, 10_000_000, 100_000, 10_000)
    txn_cost_bps= st.slider("Transaction Cost (bps)", 0, 30, 5, 1)

    run_btn = st.button("▶  RUN ANALYSIS", use_container_width=True)


# ══════════════════════════════════════════════════════════════════
#  MAIN HEADER
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<div style='padding: 24px 0 16px 0;'>
  <div style='font-family: Space Mono, monospace; font-size: 1.8rem; 
              color: #f8fafc; font-weight: 700; letter-spacing: -0.01em;'>
    Statistical Pairs Trading Engine
  </div>
  <div style='font-size: 0.88rem; color: #64748b; margin-top: 6px;'>
    OLS Hedge Ratio  ·  ADF Cointegration Test  ·  EWMA Z-Score Signals  ·  Walk-Forward Backtest
  </div>
</div>
<hr style='border: none; border-top: 1px solid #1e2d4a; margin: 0 0 20px 0;'/>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════

if "results" not in st.session_state:
    st.session_state.results = None

if run_btn or st.session_state.results is None:
    with st.spinner("Computing…"):
        if "Synthetic" in data_mode:
            df, pair_label = generate_synthetic_pair(
                n=n_days, beta=true_beta, alpha=true_alpha,
                noise_std=noise_std, drift=drift_val, seed=int(sim_seed))
        else:
            df, pair_label = fetch_real_data(ticker_x, ticker_y, period)
            if df is None:
                st.error(f"❌ Could not fetch data: {pair_label}. Make sure yfinance is installed (`pip install yfinance`) and tickers are valid.")
                st.stop()

        engine = PairsMathEngine()
        beta, alpha_ols, ols_model = engine.compute_hedge_ratio(df["Asset_Y"], df["Asset_X"])
        spread   = engine.compute_spread(df["Asset_Y"], df["Asset_X"], beta, alpha_ols)
        adf      = engine.adf_test(spread)
        eg       = engine.engle_granger_coint(df["Asset_Y"], df["Asset_X"])
        ewma_z   = engine.ewma_zscore(spread, halflife)
        roll_z   = engine.rolling_zscore(spread, roll_window)
        signals  = engine.generate_signals(ewma_z, entry_z, exit_z)
        tc_per_unit = txn_cost_bps / 10000
        returns  = engine.compute_returns(spread, signals) - abs(signals.diff()) * tc_per_unit * spread.abs()
        perf     = engine.performance_metrics(returns)

        st.session_state.results = {
            "df": df, "pair_label": pair_label,
            "beta": beta, "alpha_ols": alpha_ols, "ols_model": ols_model,
            "spread": spread, "adf": adf, "eg": eg,
            "ewma_z": ewma_z, "roll_z": roll_z,
            "signals": signals, "returns": returns, "perf": perf,
            "roll_window": roll_window,
        }


r = st.session_state.results
if r is None:
    st.info("Configure parameters in the sidebar and click **▶ RUN ANALYSIS**.")
    st.stop()


# ══════════════════════════════════════════════════════════════════
#  SIGNAL BANNER
# ══════════════════════════════════════════════════════════════════

latest_z = r["ewma_z"].dropna().iloc[-1]
latest_s = r["signals"].iloc[-1]
adf_p_thresh = {"1%": 0.01, "5%": 0.05, "10%": 0.10}
adf_crit_key = {"1%": "critical_1", "5%": "critical_5", "10%": "critical_10"}
stationary   = r["adf"]["p_value"] < adf_p_thresh[adf_sig]
adf_crit     = r["adf"][adf_crit_key[adf_sig]]
if not stationary:
    banner_cls = "signal-neutral"
    signal_txt = "⚠️  NON-STATIONARY — TRADE BLOCKED"
    signal_sub = (f"ADF p-value = {r['adf']['p_value']:.4f} &gt; 0.05 · "
                  f"Structural relationship may be broken")
elif latest_s == 1:
    banner_cls = "signal-buy"
    signal_txt = "🟢  LONG SPREAD  (Buy Y · Sell X)"
    signal_sub = (f"Z-Score = {latest_z:.3f}σ &lt; -{entry_z}σ · "
                  f"Spread is historically cheap — mean reversion expected")
elif latest_s == -1:
    banner_cls = "signal-sell"
    signal_txt = "🔴  SHORT SPREAD  (Sell Y · Buy X)"
    signal_sub = (f"Z-Score = {latest_z:.3f}σ &gt; +{entry_z}σ · "
                  f"Spread is historically expensive — convergence expected")
else:
    banner_cls = "signal-neutral"
    signal_txt = "⚪  NEUTRAL — No Active Position"
    signal_sub = f"Z-Score = {latest_z:.3f}σ · Inside ±{entry_z}σ threshold"

st.markdown(f"""
<div class="{banner_cls}">
  <div class="signal-text">{signal_txt}</div>
  <div class="signal-sub">{signal_sub}</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  KPI ROW
# ══════════════════════════════════════════════════════════════════

perf = r["perf"]
n_trades = int((r["signals"].diff().abs() > 0).sum())
n_long   = int((r["signals"] == 1).sum())
n_short  = int((r["signals"] ==-1).sum())

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Pair", r["pair_label"][:14] + ("…" if len(r["pair_label"]) > 14 else ""))
k2.metric("Hedge Ratio β", f"{r['beta']:.4f}")
k3.metric("OLS R²", f"{r['ols_model'].rsquared:.4f}")
k4.metric("ADF p-value", f"{r['adf']['p_value']:.4f}",
          delta="Stationary ✓" if stationary else "⚠ Non-Stationary",
          delta_color="normal" if stationary else "inverse")
k5.metric("Sharpe Ratio", f"{perf.get('sharpe', 0):.2f}")
k6.metric("Max Drawdown", f"{perf.get('max_drawdown', 0)*100:.2f}%",
          delta_color="inverse")


# ══════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Price & Spread",
    "⚡ Z-Score & Signals",
    "📐 Regression",
    "💰 Backtest",
    "🔬 Statistics",
])


# ── Tab 1: Price & Spread ──────────────────────────────────────
with tab1:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(chart_prices(r["df"], r["pair_label"]),
                        use_container_width=True, config={"displayModeBar": False})
    with c2:
        st.plotly_chart(chart_rolling_correlation(r["df"], r["roll_window"]),
                        use_container_width=True, config={"displayModeBar": False})
    st.plotly_chart(chart_spread(r["spread"], r["beta"]),
                    use_container_width=True, config={"displayModeBar": False})


# ── Tab 2: Z-Score & Signals ───────────────────────────────────
with tab2:
    st.plotly_chart(chart_zscore(r["ewma_z"], entry_z, r["signals"]),
                    use_container_width=True, config={"displayModeBar": False})
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_zscore_dist(r["ewma_z"]),
                        use_container_width=True, config={"displayModeBar": False})
    with c2:
        # EWMA vs Rolling comparison
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(x=r["ewma_z"].index, y=r["ewma_z"],
                                      name="EWMA Z-Score",
                                      line=dict(color="#fb923c", width=1.5)))
        fig_cmp.add_trace(go.Scatter(x=r["roll_z"].index, y=r["roll_z"],
                                      name="Rolling Z-Score",
                                      line=dict(color="#94a3b8", width=1.2, dash="dot")))
        fig_cmp.update_layout(**LAYOUT_BASE,
                               title=dict(text="EWMA vs Rolling Z-Score",
                                          font=dict(family="Space Mono", size=13, color="#e2e8f0")))
        st.plotly_chart(fig_cmp, use_container_width=True, config={"displayModeBar": False})


# ── Tab 3: Regression ─────────────────────────────────────────
with tab3:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(chart_regression(r["df"]["Asset_Y"], r["df"]["Asset_X"],
                                          r["beta"], r["alpha_ols"]),
                        use_container_width=True, config={"displayModeBar": False})
    with c2:
        st.markdown('<div class="section-header">OLS SUMMARY</div>', unsafe_allow_html=True)
        mdl = r["ols_model"]
        stats_pairs = [
            ("β (Hedge Ratio)", f"{r['beta']:.6f}", None),
            ("α (Intercept)",   f"{r['alpha_ols']:.4f}", None),
            ("R²",              f"{mdl.rsquared:.6f}", None),
            ("Adj. R²",         f"{mdl.rsquared_adj:.6f}", None),
            ("F-Statistic",     f"{mdl.fvalue:.2f}", None),
            ("F p-value",       f"{mdl.f_pvalue:.2e}", None),
            ("AIC",             f"{mdl.aic:.2f}", None),
            ("BIC",             f"{mdl.bic:.2f}", None),
            ("Observations",    str(int(mdl.nobs)), None),
        ]
        for label, val, _ in stats_pairs:
            st.markdown(f"""
            <div class="stat-row">
              <span class="stat-label">{label}</span>
              <span class="stat-value">{val}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:20px">RESIDUAL DIAGNOSTICS</div>',
                    unsafe_allow_html=True)
        resid = mdl.resid
        jb_result = stats.jarque_bera(resid)
        jb_stat, jb_p = jb_result.statistic, jb_result.pvalue
        skew = float(stats.skew(resid))
        kurt = float(stats.kurtosis(resid))   # excess kurtosis
        _, sw_p = stats.shapiro(resid[:500])
        for label, val, cls in [
            ("Skewness",       f"{skew:.4f}", "warning" if abs(skew) > 0.5 else "positive"),
            ("Excess Kurtosis",f"{kurt:.4f}", "warning" if kurt > 3 else "positive"),
            ("Jarque-Bera p",  f"{jb_p:.4f}", "negative" if jb_p < 0.05 else "positive"),
            ("Shapiro-Wilk p", f"{sw_p:.4f}", "negative" if sw_p < 0.05 else "positive"),
        ]:
            st.markdown(f"""
            <div class="stat-row">
              <span class="stat-label">{label}</span>
              <span class="stat-value {cls}">{val}</span>
            </div>""", unsafe_allow_html=True)


# ── Tab 4: Backtest ────────────────────────────────────────────
with tab4:
    st.plotly_chart(chart_equity(r["returns"]),
                    use_container_width=True, config={"displayModeBar": False})

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="section-header">RETURNS</div>', unsafe_allow_html=True)
        for label, val, cls in [
            ("Total P&L",    f"${perf.get('total_return',0)*capital:,.0f}",
             "positive" if perf.get('total_return',0) > 0 else "negative"),
            ("Ann. Return",  f"{perf.get('ann_return',0)*100:.2f}%",
             "positive" if perf.get('ann_return',0) > 0 else "negative"),
            ("Ann. Vol",     f"{perf.get('ann_vol',0)*100:.2f}%", None),
            ("Sharpe Ratio", f"{perf.get('sharpe',0):.3f}",
             "positive" if perf.get('sharpe',0) > 1 else ("warning" if perf.get('sharpe',0) > 0 else "negative")),
        ]:
            st.markdown(f"""
            <div class="stat-row">
              <span class="stat-label">{label}</span>
              <span class="stat-value {cls or ''}">{val}</span>
            </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-header">RISK</div>', unsafe_allow_html=True)
        for label, val, cls in [
            ("Max Drawdown", f"{perf.get('max_drawdown',0)*100:.2f}%", "negative"),
            ("Win Rate",     f"{perf.get('win_rate',0)*100:.1f}%",
             "positive" if perf.get('win_rate',0) > 0.5 else "warning"),
            ("# Signals",   str(n_trades), None),
            ("Long / Short", f"{n_long} / {n_short}", None),
        ]:
            st.markdown(f"""
            <div class="stat-row">
              <span class="stat-label">{label}</span>
              <span class="stat-value {cls or ''}">{val}</span>
            </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="section-header">PARAMETERS</div>', unsafe_allow_html=True)
        for label, val in [
            ("EWMA Half-Life", f"{halflife}d"),
            ("Entry |z| >",    f"{entry_z}σ"),
            ("Exit  |z| <",    f"{exit_z}σ"),
            ("Txn Cost",       f"{txn_cost_bps}bps"),
        ]:
            st.markdown(f"""
            <div class="stat-row">
              <span class="stat-label">{label}</span>
              <span class="stat-value">{val}</span>
            </div>""", unsafe_allow_html=True)


# ── Tab 5: Statistics ──────────────────────────────────────────
with tab5:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">AUGMENTED DICKEY-FULLER TEST</div>',
                    unsafe_allow_html=True)
        adf = r["adf"]
        for label, val, cls in [
            ("ADF Statistic",  f"{adf['adf_stat']:.6f}",
             "positive" if adf['adf_stat'] < adf['critical_5'] else "negative"),
            ("p-value",        f"{adf['p_value']:.6f}",
             "positive" if adf['p_value'] < 0.05 else "negative"),
            ("Lags Used",      str(adf['lags_used']), None),
            ("Observations",   str(adf['n_obs']), None),
            ("Critical 1%",    f"{adf['critical_1']:.4f}", None),
            ("Critical 5%",    f"{adf['critical_5']:.4f}", None),
            ("Critical 10%",   f"{adf['critical_10']:.4f}", None),
            ("Verdict",
             "✅ Stationary (cointegrated)" if adf['stationary'] else "❌ Non-Stationary",
             "positive" if adf['stationary'] else "negative"),
        ]:
            st.markdown(f"""
            <div class="stat-row">
              <span class="stat-label">{label}</span>
              <span class="stat-value {cls or ''}">{val}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:20px">ENGLE-GRANGER TEST</div>',
                    unsafe_allow_html=True)
        eg = r["eg"]
        for label, val, cls in [
            ("t-Statistic",  f"{eg['t_stat']:.6f}", None),
            ("p-value",      f"{eg['p_value']:.6f}",
             "positive" if eg['p_value'] < 0.05 else "negative"),
            ("Critical 1%",  f"{eg['critical_1']:.4f}", None),
            ("Critical 5%",  f"{eg['critical_5']:.4f}", None),
            ("Verdict",
             "✅ Cointegrated" if eg['cointegrated'] else "❌ Not Cointegrated",
             "positive" if eg['cointegrated'] else "negative"),
        ]:
            st.markdown(f"""
            <div class="stat-row">
              <span class="stat-label">{label}</span>
              <span class="stat-value {cls or ''}">{val}</span>
            </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-header">SPREAD DIAGNOSTICS</div>',
                    unsafe_allow_html=True)
        sp = r["spread"]
        for label, val in [
            ("Mean",       f"{sp.mean():.4f}"),
            ("Std Dev",    f"{sp.std():.4f}"),
            ("Min",        f"{sp.min():.4f}"),
            ("Max",        f"{sp.max():.4f}"),
            ("Skewness",   f"{sp.skew():.4f}"),
            ("Kurtosis",   f"{sp.kurt():.4f}"),
            ("Half-Life (est.)", f"{halflife}d (user-set)"),
        ]:
            st.markdown(f"""
            <div class="stat-row">
              <span class="stat-label">{label}</span>
              <span class="stat-value">{val}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:20px">Z-SCORE DIAGNOSTICS</div>',
                    unsafe_allow_html=True)
        zc = r["ewma_z"].dropna()
        pct_in_band = ((zc.abs() < entry_z).sum() / len(zc) * 100)
        for label, val, cls in [
            ("Current Z",  f"{latest_z:.4f}σ",
             "positive" if abs(latest_z) < entry_z else "negative"),
            ("Mean Z",     f"{zc.mean():.4f}", None),
            ("Std Z",      f"{zc.std():.4f}", None),
            ("Max |Z|",    f"{zc.abs().max():.4f}σ", None),
            ("% In Band",  f"{pct_in_band:.1f}%",
             "positive" if pct_in_band > 70 else "warning"),
            ("Signal Days", f"{(r['signals'] != 0).sum()} / {len(r['signals'])}",  None),
        ]:
            st.markdown(f"""
            <div class="stat-row">
              <span class="stat-label">{label}</span>
              <span class="stat-value {cls or ''}">{val}</span>
            </div>""", unsafe_allow_html=True)

        # Spread ACF mini-chart
        from statsmodels.stats.stattools import durbin_watson
        dw = durbin_watson(r["spread"].dropna())
        st.markdown(f"""
        <div class="stat-row" style="margin-top:16px">
          <span class="stat-label">Durbin-Watson</span>
          <span class="stat-value {'positive' if 1.5 < dw < 2.5 else 'warning'}">{dw:.4f}</span>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<hr style='border: none; border-top: 1px solid #1e2d4a; margin: 32px 0 12px 0;'/>
<div style='font-size:0.72rem; color:#334155; text-align:center; 
            font-family: Space Mono, monospace; letter-spacing:0.05em;'>
  PAIRS TRADING ENGINE · OLS · ADF · EWMA · FOR EDUCATIONAL & RESEARCH USE ONLY · NOT FINANCIAL ADVICE
</div>
""", unsafe_allow_html=True)
