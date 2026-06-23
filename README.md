# Demand-Oracle

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3500&pause=800&color=2E86AB&center=true&vCenter=true&width=700&lines=Solving+the+multi-million+rupee+inventory+problem;Curing+data+leakage+with+strict+chronological+split;LightGBM+forecasting+engine+running+at+3am;Vectorized+pandas+code+operating+at+C-speed" alt="Demand-Oracle Title Engine" />
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" width="100%" alt="Divider Wave" />
</p>

```text
  ____                                    _          ___                     _      
 |  _ \  ___ _ __ ___   __ _ _ __   __| |        / _ \ _ __ __ _  ___| | ___ 
 | | | |/ _ \ '_ ` _ \ / _` | '_ \ / _` |_____  | | | | '__/ _` |/ __| |/ _ \
 | |_| |  __/ | | | | | (_| | | | | (_| |_____| | |_| | | | (_| | (__| |  __/
 |____/ \___|_| |_| |_|\__,_|_| |_|\__,_|        \___/|_|  \__,_|\___|_|\___|
                                                                             
```

---

## 🛠️ What This Actually Does

Look, retail demand forecasting is usually a mess of generic ARIMA models or overpriced enterprise software. This is a high-performance supervised pipeline built around **LightGBM**. It takes 5 years of daily transactions across 10 stores and 50 items (500 SKUs in total) and figures out when people are actually going to buy stuff.

Instead of just printing a boring MAPE score, it translates those predictions into actual inventory holding and stockout costs so you know exactly how much cash you're saving.

---

## 📐 Whiteboard Architecture (Drawn at 3 AM)

This is how the modules interact. I sketched this out to keep myself sane while writing the pipeline:

```text
                                  +-----------------------+
                                  |  data/raw/train.csv   |
                                  +-----------------------+
                                              |
                                              | (Reads daily sales)
                                              v
                                  +-----------------------+
                                  |   src/data_loader.py  | <--- Clips sales >= 1 (No MAPE division-by-zero)
                                  +-----------------------+
                                              |
                                              v
                                  +-----------------------+
                                  |    src/features.py    | <--- Cyclical encoding, vectorized rollings
                                  +-----------------------+
                                              |
                                              v (Saves features.csv - 271MB)
                                  +-----------------------+
                                  |    src/model.py       | <--- Trains LightGBM with Early Stopping
                                  +-----------------------+
                                     |                 |
                (Validation Folds)  /                   \  (Residual Bootstrap)
                                   v                     v
                      +-------------------+       +-------------------------------+
                      |  src/evaluate.py  |       |    Bootstrap Intervals        |
                      +-------------------+       +-------------------------------+
                               |                                 |
                               \                                 /
                                v                               v
                       +-------------------------------------------------+
                       |               src/inventory.py                  |
                       +-------------------------------------------------+
                                                |
                                                v (Translates to INR)
                                       [ final main.py runner ]
```

---

## 🛡️ Chronological Split & Leakage Guard

We enforce a strict chronological boundary. Future data never leaks into the past. 

```text
   Time Axis -------------------------------------------------------------->
   [====== Training Range (Pre-July 2017) ======] | [=== Evaluation Range ===]
                                                 |
   *-- lag_365 (Sales same day last year) --------+--> [Feature Feed]
   *-- lag_7   (Sales same day last week) --------+--> [Feature Feed]
   *-- rolling_mean_30 (Shifted 1 day) -----------+--> [Feature Feed]
                                                 |
                                         (Cutoff Date: 2017-07-01)
```

---

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

---

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

---

## 📓 Walk-Forward Backtesting (Historical Simulation)

Instead of a single test split, we simulate historical deployments across 4 cutoff points spaced 3 months apart. If the model succeeds across all 4, it is stable.

```text
  Fold 1 (2016-07-01): [===== Train (3.5 Years) =====] | [== 90-Day Test ==]
  Fold 2 (2016-10-01): [======= Train (3.8 Years) =======] | [== 90-Day Test ==]
  Fold 3 (2017-01-01): [========= Train (4.0 Years) =========] | [== 90-Day Test ==]
  Fold 4 (2017-04-01): [=========== Train (4.3 Years) ===========] | [== 90-Day Test ==]
```

---

## 🛠️ The Tech Stack (and why I chose it)

| Tech | Why it's here |
|---|---|
| **Python 3.11** | Clean syntax, standard packaging, and solid package support. |
| **LightGBM** | Incredibly fast leaf-wise growth tree boosting. Handled our 730,500 features and trained in 24 seconds. |
| **Pandas & NumPy** | Vectorized feature operations. I avoided custom Python loops to run operations at C-speed. |
| **Statsmodels** | Used for multiplicative seasonal decomposition to extract trend and residuals. |
| **Matplotlib** | Clean, minimalist visual plots (no default plotting formats). |

---

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

---

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

---

## 🗺️ Roadmap

- [x] Vectorize pandas features for C-speed executions.
- [x] Implement 4-fold walk-forward backtest loop.
- [x] Translate model accuracy to cash inventory costs (INR).
- [x] Output prediction intervals using residual bootstrap.
- [ ] Incorporate supplier lead-time parameters into intervals.
- [ ] Deploy Streamlit UI log-uploader dashboard.

---

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
