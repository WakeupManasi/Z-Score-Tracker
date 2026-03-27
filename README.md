# ⚡ Statistical Pairs Trading Engine

A rigorous, portfolio-grade statistical arbitrage system built with Python.  
Implements OLS hedge ratio estimation, ADF cointegration testing, and EWMA Z-Score signal generation with a reactive Streamlit UI.

---

## 🏗️ Architecture

```
pairs_trading_engine/
├── app.py               # Full application (math engine + UI)
├── requirements.txt     # Python dependencies
└── README.md
```

### Core Components

| Layer | Component | Purpose |
|---|---|---|
| **Math** | `PairsMathEngine` | OLS regression, ADF test, EWMA Z-score, signal generation, backtest |
| **Data** | `generate_synthetic_pair` / `fetch_real_data` | Ornstein-Uhlenbeck simulation or live yfinance data |
| **UI** | Streamlit + Plotly | Reactive charts, parameter sliders, signal banners |

---

## 📐 The Mathematics

### 1. OLS Hedge Ratio
Regresses Asset Y on Asset X to find the exact hedge ratio β:

```
Y = α + β·X + ε
```

β tells us: "for every $1 move in X, Y should move $β" — neutralizing market beta.

### 2. The Spread
```
Spread_t = Y_t - (β·X_t + α)
```
This is the *mean-reverting residual* — the stationary series we trade.

### 3. ADF Stationarity Test
The Augmented Dickey-Fuller test validates:
- **H₀**: Spread has a unit root (non-stationary, DO NOT trade)
- **H₁**: Spread is stationary (safe to trade mean reversion)

If p-value > 0.05, the engine **blocks** the trade signal.

### 4. EWMA Z-Score
```
μ_t  = EWMA(Spread, halflife)
σ_t  = EWMA_std(Spread, halflife)
Z_t  = (Spread_t - μ_t) / σ_t
```

Signal logic:
- Z > +2σ → **SHORT spread** (sell Y, buy X) — spread will narrow
- Z < -2σ → **LONG spread** (buy Y, sell X) — spread will widen
- |Z| < 0.5σ → **CLOSE position** — mean reversion achieved

---

## 🚀 Setup

```bash
# 1. Clone / copy files to your directory
cd pairs_trading_engine

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Open browser at **http://localhost:8501**

---

## 📡 Live Market Data

Install `yfinance` (included in requirements.txt) and switch to **Live Market Data** mode.

Suggested cointegrated pairs:
| Sector | Pair |
|---|---|
| Beverages | KO / PEP |
| Investment Banks | GS / MS |
| Energy Majors | XOM / CVX |
| Airlines | DAL / UAL |
| Gold Miners | GLD / GDX |
| Big Tech | MSFT / GOOGL |

---

## ⚙️ Parameters

| Parameter | Description | Default |
|---|---|---|
| `EWMA Half-Life` | Decay speed of exponential weights (days) | 30 |
| `Rolling Window` | Window for rolling z-score & correlation | 60 |
| `Entry Z-Score` | Threshold to open a position | 2.0σ |
| `Exit Z-Score` | Threshold to close a position | 0.5σ |
| `ADF Significance` | Significance level for stationarity | 5% |
| `Transaction Cost` | Round-trip cost in basis points | 5bps |

--📉 Backtesting Logic
## 📉 Backtesting Logic

The engine simulates trading performance using:

- Entry/exit based on Z-score thresholds
- Position sizing based on hedge ratio
- Transaction cost modeling (bps)
- Cumulative PnL tracking

### Performance Metrics

- Total Return
- Sharpe Ratio (optional extension)
- Max Drawdown
- Win Rate
-
This project demonstrates a practical implementation of statistical arbitrage using robust quantitative methods. 
It combines financial theory with real-time visualization to create an intuitive trading research tool.

## ⚠️ Disclaimer

This software is for **educational and research purposes only**. It does not constitute financial advice. Past performance of any backtested strategy is not indicative of future results. Always consult a qualified financial professional before trading.

---

*Built with Python · pandas · numpy · statsmodels · plotly · streamlit*
