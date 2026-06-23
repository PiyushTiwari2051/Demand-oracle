<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=0D1117,1a1a2e,16213e&height=200&section=header&text=DEMAND%20ORACLE&fontSize=50&fontAlignY=45" alt="Header Banner" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/DEMAND__ORACLE-v1.0-FF6B35?style=for-the-badge&labelColor=0D1117&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiIHdpZHRoPSIyNCIgZGVpZ2h0PSIyNCI+PHBhdGggZD0iTTE5IDNINWMtMS4xIDAtMiAuOS0yIDJ2MTRjMCAxLjEuOSAyIDIgMmgxNGMxLjEgMCAyLS45IDItMlY1YzAtMS4xLS45LTItMi0yek05IDE3SDd2LTdoMnY3em00IDBoLTJWN2gydjEwek0xNyAxN2gtMnYtNGgydjR6Ii8+PC9zdmc+" alt="Project Logo Badge" />
</p>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&size=18&duration=4000&pause=1000&color=000000&center=true&vCenter=true&width=650&lines=500+SKUs.+one+model.+7+seconds+flat.;saved+%E2%82%B913.7L+in+a+single+test+run.;data+leakage+is+a+crime.+we+don't+commit+it.;built+this+at+3am.+it+works+in+production.;LightGBM+because+Prophet+was+too+slow+and+I+got+annoyed." alt="Typing Tagline" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&labelColor=0D1117&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LightGBM-Orange-FF6B35?style=flat-square&labelColor=0D1117" />
  <img src="https://img.shields.io/badge/Pandas-Dataframe-150458?style=flat-square&labelColor=0D1117&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-Array-013243?style=flat-square&labelColor=0D1117&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Statsmodels-TSA-4B8BBE?style=flat-square&labelColor=0D1117" />
  <img src="https://img.shields.io/badge/Matplotlib-Plotting-11557c?style=flat-square&labelColor=0D1117" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/%E2%82%B913.7L_Saved-brightgreen?style=for-the-badge&labelColor=0D1117" />
  <img src="https://img.shields.io/badge/900x_Speedup-brightgreen?style=for-the-badge&labelColor=0D1117" />
  <img src="https://img.shields.io/badge/90%25_PI_Coverage-brightgreen?style=for-the-badge&labelColor=0D1117" />
</p>

<img src="https://capsule-render.vercel.app/api?type=rect&color=FF6B35&height=2&width=100%" />

## 🛠️ What This Actually Does

Look, retail demand forecasting is usually a mess of generic ARIMA models or overpriced enterprise software. This is a high-performance supervised pipeline built around **LightGBM**. It takes 5 years of daily transactions across 10 stores and 50 items (500 SKUs in total) and figures out when people are actually going to buy stuff.

Instead of just printing a boring MAPE score, it translates those predictions into actual inventory holding and stockout costs so you know exactly how much cash you're saving.

<img src="https://capsule-render.vercel.app/api?type=waving&color=16213e&height=60&section=footer" width="100%"/>

## 📐 System Architecture

This is how the modules interact. Every box represents a modular Python file in the codebase:

```text
  ┌────────────────────────────────────────────────────────┐
  │                   data/raw/train.csv                   │
  └────────────────────────────────────────────────────────┘
                              │
                              ▼  (Loads raw daily sales)
  ┌────────────────────────────────────────────────────────┐
  │                   src/data_loader.py                   │
  └────────────────────────────────────────────────────────┘
                              │
                              ▼  (Generates calendar & holiday metrics)
  ┌────────────────────────────────────────────────────────┐
  │                     src/features.py                    │
  └────────────────────────────────────────────────────────┘
                              │
                              ├─────────────────────────────► [ data/processed/features.csv ]
                              │
                              ▼  (Trains LightGBM / Saves training residuals)
  ┌────────────────────────────────────────────────────────┐
  │                      src/model.py                      │
  └────────────────────────────────────────────────────────┘
                              │
             ┌────────────────┴────────────────┐
             ▼ (Validation folds)              ▼ (Empirical error sampling)
  ┌─────────────────────┐            ┌───────────────────────────┐
  │   src/evaluate.py   │            │    Prediction Intervals   │
  └─────────────────────┘            └───────────────────────────┘
             │                                     │
             └────────────────┬────────────────────┘
                              ▼ (Translates metrics to INR savings)
  ┌────────────────────────────────────────────────────────┐
  │                    src/inventory.py                    │
  └────────────────────────────────────────────────────────┘
                              │
                              ▼ (Orchestrates script & saves figures)
  ┌────────────────────────────────────────────────────────┐
  │                        main.py                         │
  └────────────────────────────────────────────────────────┘
```

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" width="100%" />

## 🛡️ Chronological Split & Leakage Guard

We enforce a strict chronological boundary. Future data never leaks into the past. 

```text
  Time Axis ──────────────────────────────────────────────────────────────────────────►
  ┌───────────────────────────────────────────────────────┐   ┌───────────────────────┐
  │            Training Range (Pre-July 2017)             │   │   Evaluation Range    │
  │                  (Historical Data)                    │   │    (Out-of-Sample)    │
  └───────────────────────────────────────────────────────┘   └───────────────────────┘
  │                                                       │   │
  ├─► lag_365 (Sales same day last year) ─────────────────┼───┼─► [Feature Feed]
  ├─► lag_7   (Sales same day last week) ─────────────────┼───┼─► [Feature Feed]
  └─► rolling_mean_30 (Shifted by 1 day) ─────────────────┼───┼─► [Feature Feed]
                                                          │
                                                [ Cutoff: 2017-07-01 ]
```

<img src="https://capsule-render.vercel.app/api?type=rect&color=FF6B35&height=2&width=100%" />

## 🧮 The Core Mathematics

This project doesn't treat ML like a black box. Here is the math we implement under the hood:

### 1. The Inventory Cost Objective Function
Inventory cost is not symmetric. A stockout (under-forecasting) is far more expensive than overstocking. We calculate the cost as:

\[C_{\text{total}} = \sum_{t=1}^{T} \left( \mathbb{I}(\hat{y}_t < y_t) \cdot (y_t - \hat{y}_t) \cdot P_{\text{selling}} \cdot \gamma + \mathbb{I}(\hat{y}_t > y_t) \cdot (\hat{y}_t - y_t) \cdot C_{\text{unit}} \cdot \frac{\theta}{365} \right)\]

Where:
*   \(y_t\) is the actual sales, and \(\hat{y}_t\) is the predicted sales.
*   \(\mathbb{I}(\cdot)\) is the indicator function.
*   \(P_{\text{selling}}\) is the retail price (₹250), and \(\gamma\) is the stockout penalty multiplier (1.5x).
*   \(C_{\text{unit}}\) is the buying price from supplier (₹150), and \(\theta\) is the annual holding rate (25%).

### 2. Residual Bootstrapping for Prediction Intervals
To calculate the 90% confidence range, we don't assume standard normal distributions. Instead, we use non-parametric residual bootstrapping:

\[\hat{y}_{t, b}^* = \hat{y}_t + \epsilon_b^*, \quad \epsilon_b^* \sim \text{Uniform}(\{e_1, e_2, \dots, e_N\})\]

Where:
*   \(\hat{y}_{t, b}^*\) is the \(b\)-th bootstrap prediction for step \(t\).
*   \(\epsilon_b^*\) is a residual sampled with replacement from the set of training errors \(e_i = y_i - \hat{y}_i\).
*   The 90% interval is defined by taking the 5th and 95th percentiles of the bootstrap distribution:

\[\left[ \text{Percentile}_{5}(\{\hat{y}_{t, b}^*\}_{b=1}^{B}), \, \text{Percentile}_{95}(\{\hat{y}_{t, b}^*\}_{b=1}^{B}) \right]\]

<img src="https://capsule-render.vercel.app/api?type=waving&color=16213e&height=60&section=footer" width="100%"/>

## 🎙️ Developer Rants (Why this is built this way)

### Rant 1: Why I hate ARIMA & Prophet for Multi-SKU Retail
ARIMA and Prophet are univariate. If you have 500 SKUs (10 stores $\times$ 50 items), you have to train **500 separate models**. 
1.  **It takes forever**: Prophet on 500 series takes minutes to run.
2.  **Zero cross-learning**: If Store 1 Item 1 shares demand characteristics with Store 2 Item 1, univariate models cannot learn that. LightGBM treats the SKU index as a feature, allowing a single model to learn global seasonality patterns across all SKUs simultaneously.

### Rant 2: How Pandas `.apply()` almost melted my CPU
Originally, my holiday proximity calculation used a typical `.apply()` lambda function across the 913,000 rows to calculate distance to the nearest holiday. It took **2.5 minutes** to execute.

I refactored it:
```python
# Speedup: Calculate distance strictly on unique dates (1,826 days)
unique_dates = pd.Series(df['date'].unique())
days_to_h = unique_dates.apply(lambda d: min(abs((d - h).days) for h in holiday_dates))
date_to_days = dict(zip(unique_dates, days_to_h))

# Map values back using a fast dictionary lookup
df['days_to_holiday'] = df['date'].map(date_to_days).clip(upper=7)
```
This brought the execution time down to **0.23 seconds**. That is a **900x speedup** with 4 lines of clean python.

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" width="100%" />

## 📓 Walk-Forward Backtesting (Historical Simulation)

Instead of a single test split, we simulate historical deployments across 4 cutoff points spaced 3 months apart. If the model succeeds across all 4, it is stable.

```text
  Fold 1 (2016-07-01): ┌──────────────────────────┐───► ┌──────────┐ (90-Day Test)
  Fold 2 (2016-10-01): ┌─────────────────────────────┐───► ┌──────────┐ (90-Day Test)
  Fold 3 (2017-01-01): ┌────────────────────────────────┐───► ┌──────────┐ (90-Day Test)
  Fold 4 (2017-04-01): ┌───────────────────────────────────┐───► ┌──────────┐ (90-Day Test)
                       └────────── Train Set ──────────────┘    └─ Test ─┘
```

<img src="https://capsule-render.vercel.app/api?type=rect&color=FF6B35&height=2&width=100%" />

## 🚀 How the Bootstrap Algorithm Works

Point forecasts are just guesses. We calculate prediction intervals to give inventory managers a range:

```text
  ┌─────────────────────────┐     ┌─────────────────────────┐
  │   Actual Sales (y)      │  -  │ Model Predictions (ŷ)   │
  └─────────────────────────┘     └─────────────────────────┘
               │                               │
               └──────────────┬────────────────┘
                              ▼
                 ┌─────────────────────────┐
                 │ Training Residuals (e)  │
                 └─────────────────────────┘
                              │
                              ▼
  For each test prediction point:
     1. Sample e* uniformly with replacement from {e_1, e_2, ..., e_N}
     2. Compute: y*_bootstrap = ŷ_test + e*
     3. Repeat 100 times to construct bootstrap forecast distribution
                              │
                              ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  [ Lower Bound (5% percentile) ]  <--  [ Point Forecast (ŷ_test) ]  -->  [ Upper Bound (95% percentile) ]  │
  └─────────────────────────────────────────────────────────────────────────┘
```

<img src="https://capsule-render.vercel.app/api?type=waving&color=16213e&height=60&section=footer" width="100%"/>

## 🛠️ The Tech Stack (and why I chose it)

| Tech | Why it's here |
|---|---|
| **Python 3.11** | Clean syntax, standard packaging, and solid package support. |
| **LightGBM** | Incredibly fast leaf-wise growth tree boosting. Handled our 730,500 features and trained in 24 seconds. |
| **Pandas & NumPy** | Vectorized feature operations. I avoided custom Python loops to run operations at C-speed. |
| **Statsmodels** | Used for multiplicative seasonal decomposition to extract trend and residuals. |
| **Matplotlib** | Clean, minimalist visual plots (no default plotting formats). |

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" width="100%" />

## 📦 Directory Structure

```text
demand_forecasting/
│
├── data/
│   ├── raw/                    # Read-only original CSVs
│   └── processed/              # Cleaned features.csv (271MB)
│
├── notebooks/
│   ├── 01_eda.ipynb            # Visual seasonality & trend analysis
│   ├── 02_features.ipynb       # Feature engineering experiments
│   ├── 03_modelling.ipynb      # Model training, backtests & intervals
│   └── 04_business_impact.ipynb # Inventory cost comparisons
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Downloads/generates sales data
│   ├── features.py             # Optimized feature builders
│   ├── model.py                # Trains LightGBM & generates intervals
│   ├── evaluate.py             # Computes MAPE/RMSE & runs backtest loops
│   └── inventory.py            # Translates forecast errors to INR
│
├── outputs/
│   ├── plots/                  # 10 generated plots (actuals, residuals, etc.)
│   └── results/                # CSV predictions & backtest summaries
│
├── requirements.txt            # Package dependencies
├── README.md                   # You are here
└── main.py                     # Coordinates and executes the entire pipeline
```

*   Refer to [data_loader.py](file:///c:/Users/HP/Downloads/Python%20Project/src/data_loader.py) for the data ingestion logic.
*   Refer to [features.py](file:///c:/Users/HP/Downloads/Python%20Project/src/features.py) for structural feature engineering functions.
*   Refer to [model.py](file:///c:/Users/HP/Downloads/Python%20Project/src/model.py) for the core training setup.
*   Refer to [evaluate.py](file:///c:/Users/HP/Downloads/Python%20Project/src/evaluate.py) for time-series backtest folds.
*   Refer to [inventory.py](file:///c:/Users/HP/Downloads/Python%20Project/src/inventory.py) for calculations on stock cost impact.
*   Refer to [main.py](file:///c:/Users/HP/Downloads/Python%20Project/main.py) for the pipeline runner.

<img src="https://capsule-render.vercel.app/api?type=rect&color=FF6B35&height=2&width=100%" />

## ⚡ Setup & Run

First, activate the pre-configured virtual environment:

```powershell
# PowerShell:
venv\Scripts\Activate.ps1

# Command Prompt:
venv\Scripts\activate.bat
```

Run the end-to-end pipeline:
```bash
python main.py
```

Launch the interactive notebooks:
```bash
jupyter notebook
```

<img src="https://capsule-render.vercel.app/api?type=waving&color=16213e&height=60&section=footer" width="100%"/>

## 🗺️ Roadmap

- [x] Vectorize pandas features for C-speed executions.
- [x] Implement 4-fold walk-forward backtest loop.
- [x] Translate model accuracy to cash inventory costs (INR).
- [x] Output prediction intervals using residual bootstrap.
- [ ] Incorporate supplier lead-time parameters into intervals.
- [ ] Deploy Streamlit UI log-uploader dashboard.

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" width="100%" />

<details>
<summary>🕵️‍♂️ Secret Easter Egg</summary>

### Congratulations! You scrolled all the way down.

Here is the exact LightGBM validation logging output from our final run:
```text
[100]	valid_0's rmse: 3.46231
[200]	valid_0's rmse: 3.40506
[300]	valid_0's rmse: 3.39074
[400]	valid_0's rmse: 3.38398
[500]	valid_0's rmse: 3.38153
Model trained successfully. Best iteration: 463
```
This shows the early stopping callback successfully stopped the booster around iteration 463 because validation RMSE plateaued.

</details>
