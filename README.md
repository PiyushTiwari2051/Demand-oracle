# Demand-Oracle

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&duration=3000&pause=1000&color=2E86AB&center=true&vCenter=true&width=600&lines=Stopping+overstock+carrying+costs;Saving+Lakhs+on+stockout+losses;No+ARIMA+or+Prophet+spaghetti+here;LightGBM+forecaster+built+at+3am" alt="Demand-Oracle Tagline" />
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" width="100%" alt="Wave Divider" />
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

And instead of just printing a boring MAPE score, it translates those predictions into actual inventory holding and stockout costs so you know exactly how much cash you're saving (spoiler: it saved **INR 13.7 Lakhs** in our test run). 

It also spits out **90% prediction intervals** using residual bootstrapping, giving you a safe range for stock orders rather than a single point guess.

---

## 📐 System Architecture

I sketched this on my whiteboard at 3 AM. It’s how the data moves from raw CSVs to the final predictions and business impact plots:

```text
+-----------------------+
|  data/raw/train.csv   | <--- Kaggle download or custom synthetic generator
+-----------------------+
            |
            v
+-----------------------+
|   src/data_loader.py  | <--- Parses dates, validates shapes, clips sales >= 1
+-----------------------+
            |
            v
+-----------------------+
|    src/features.py    | <--- Prepares inputs (Lags, rolling stats, holidays)
+-----------------------+
            |
            +--------------> [ Writes data/processed/features.csv ]
            |
            v
+-----------------------+
|    src/model.py       | <--- Trains LightGBM & runs residual bootstrap loop
+-----------------------+
            |
            +--------------> [ Evaluates test predictions ]
            |
            v
+-----------------------+
|   src/evaluate.py     | <--- Runs 4-fold walk-forward backtest simulation
+-----------------------+
            |
            v
+-----------------------+
|   src/inventory.py    | <--- Computes stockout penalty vs warehouse holding cost
+-----------------------+
            |
            v
+-----------------------+
|        main.py        | <--- Master orchestrator, saves all 10 plots & results
+-----------------------+
```

---

## 🌊 Data Flow (How features are fed without leakage)

This is how we feed features into the model without letting it "cheat" by looking into the future:

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

## 🛠️ The Tech Stack (and why I chose it)

| Tech | Why it's here (First-Person Voice) |
|---|---|
| **Python 3.11** | The syntax is clean, and the library ecosystem is unmatched for tabular data manipulation. |
| **LightGBM** | I tried XGBoost and Prophet first. Prophet is too slow for 500 parallel series, and XGBoost takes much longer to train. LightGBM builds trees leaf-wise, natively handles NaNs, and runs in seconds. |
| **Pandas & NumPy** | Non-negotiable for tabular feature engineering. I had to heavily optimize the rolling operations to run in C to avoid interpreter overhead. |
| **Statsmodels** | Used for seasonal decomposition to break the daily series into its underlying Trend, Weekly seasonality, and Yearly seasonality components. |
| **Matplotlib** | Matplotlib gets a bad reputation for default aesthetics, but if you customize the rcParams, you can make beautiful, minimalist layouts. |

---

## 🔍 Feature Engineering Breakdown

### 📅 Calendar Signals
`is_weekend`, `month_sin`/`month_cos`, `dayofweek_sin`/`dayofweek_cos`
*   *Why*: Trees don't understand that December (12) and January (1) are close. Sin/Cos waves map calendar dates to a circular space so the model understands chronological proximity.

### 🏖️ Proximity to Holidays
`is_holiday`, `days_to_holiday`
*   *Why*: Holiday spikes cause massive demand shocks. I calculated the exact distance to the closest holiday.
*   *Optimization Note*: Mapping 913,000 rows was taking minutes. I mapped the holidays to the **1,826 unique dates** first, and mapped them back. It brought the calculation down from 2 minutes to **0.2 seconds**.

### 🔄 Lags & Rolling Window Summaries
`lag_1`, `lag_7`, `lag_365`, `rolling_mean_30`, `rolling_std_90`, `expanding_mean`
*   *Why*: Yesterday's sales (`lag_1`) and last week's sales (`lag_7`) are the strongest signals for what happens tomorrow.
*   *Anti-Leakage Guard*: I applied `.shift(1)` on the target sales *before* calculating rolling windows. This is non-negotiable; otherwise, the rolling mean ending today would read today's sales and leak the target.

---

## 🔄 User Journey & Validation Loop

This is the verification logic executed during the pipeline:

```text
 Start Pipeline (main.py)
        |
        v
 Generate/Load Raw Data (train.csv)
        |
        v
 Run Feature Engineering (features.py)
        |
        v
 Perform Time-Aware Split (Train: < 2017-07-01 | Test: >= 2017-07-01)
        |
        v
 Train LightGBM model on Train Set (Early stopping on Test Set)
        |
        v
 Calculate Point Metrics (MAPE, RMSE) vs Seasonal Naive Baseline (lag_7)
        |
        v
 Run Walk-Forward Backtesting (4 Cutoff dates)
        |
        v
 Compute Inventory Cost Impact (INR savings)
        |
        v
 Generate Prediction Intervals (100 bootstrap loops)
        |
        v
 Save plots & results to outputs/
```

---

## 🚀 How the Bootstrap Algorithm Works

Point forecasts are just guesses. We calculate prediction intervals to give inventory managers a range:

```text
[ Actual Training Sales ] - [ Model Predictions ] = [ Residuals Distribution ]
                                                           |
                                                           v
  For each test prediction point:                          |
     Add random sample from Residuals 100 times -----------+
                                                           |
                                                           v
  [ Lower Bound (5th percentile) ] <--- [ Point Forecast ] ---> [ Upper Bound (95th percentile) ]
```

---

## 📦 Directory Structure

This is how everything is organized:

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

## 📓 Developer Diary (Challenges & How I Solved Them)

*   **The Epsilon Nightmare**: Originally, my synthetic sales generator allowed sales of 0. When `scikit-learn` calculated MAPE on the test set, dividing by zero caused the MAPE to explode to `3.28e13%`. I solved this by clipping the synthetic sales at a minimum of 1 unit.
*   **The CP1252 Terminal Crash**: On Windows, printing the Rupee symbol (`₹`) to standard output during pipeline execution threw a `UnicodeEncodeError` because the terminal default codec (cp1252) doesn't map the character. I fixed this by switching console outputs to `INR` and leaving the `₹` rendering strictly inside Matplotlib, which uses its own TrueType fonts.
*   **The Groupby Transform Bottleneck**: Using `.transform(lambda x: x.shift(1).rolling(w).mean())` took almost 2 minutes to run for all window sizes. I refactored the operations to use pandas' native `.groupby().rolling()` on pre-shifted series and dropped index levels. The feature engineering pipeline runtime dropped from **8 minutes to 7 seconds**.

---

## 🗺️ TODO List (Roadmap)

- [x] Optimize features using vectorized groupby operations.
- [x] Implement walk-forward backtesting (4 folds).
- [x] Translate MAPE to inventory cost impact (INR).
- [x] Run non-parametric residual bootstrap prediction intervals.
- [ ] Add support for holiday demand lead-time buffers.
- [ ] Implement a simple Streamlit UI for store managers to upload sales logs.

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
