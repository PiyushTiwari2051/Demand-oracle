import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose

from src.data_loader import load_data
from src.features import generate_all_features
from src.model import train_lgb_model, predict_lgb, bootstrap_prediction_intervals, get_default_params
from src.evaluate import calculate_metrics, run_backtest
from src.inventory import calculate_inventory_impact

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# Configure matplotlib styles
plt.rcParams.update({
    'figure.figsize': (14, 5),
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.family': 'monospace',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})

def main():
    logger.info("Initializing demand forecasting pipeline...")
    
    os.makedirs('outputs/plots', exist_ok=True)
    os.makedirs('outputs/results', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # 1. Load Data
    df = load_data('data/raw/train.csv')
    
    # 2. Generate Exploratory Plots
    logger.info("Generating EDA visualization figures...")
    
    daily_total = df.groupby('date')['sales'].sum().reset_index().set_index('date').sort_index()
    
    # Plot 1: Daily total sales
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(daily_total.index, daily_total['sales'], color='#2E86AB', linewidth=0.8, alpha=0.9)
    ax.set_title('Total Daily Sales — All Stores & Items (2013–2017)', fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Units Sold')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    plt.savefig('outputs/plots/01_raw_sales_timeseries.png', dpi=150)
    plt.close()
    
    # Plot 2: Weekly seasonality
    daily_total['dayofweek'] = daily_total.index.dayofweek
    avg_by_day = daily_total.groupby('dayofweek')['sales'].mean()
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(day_names, avg_by_day.values,
                  color=['#2E86AB' if i < 5 else '#E84855' for i in range(7)],
                  edgecolor='black', linewidth=0.5)
    ax.set_title('Average Sales by Day of Week', fontweight='bold')
    ax.set_ylabel('Average Units Sold')
    for bar, val in zip(bars, avg_by_day.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'{val:,.0f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig('outputs/plots/02_weekly_pattern.png', dpi=150)
    plt.close()
    
    # Plot 3: Monthly seasonality
    daily_total['month'] = daily_total.index.month
    avg_by_month = daily_total.groupby('month')['sales'].mean()
    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(month_names, avg_by_month.values, marker='o', color='#2E86AB', linewidth=2, markersize=7)
    ax.fill_between(range(12), avg_by_month.values, alpha=0.15, color='#2E86AB')
    ax.set_title('Average Sales by Month (Yearly Seasonality)', fontweight='bold')
    ax.set_ylabel('Average Units Sold')
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_names)
    plt.tight_layout()
    plt.savefig('outputs/plots/03_monthly_pattern.png', dpi=150)
    plt.close()
    
    # Plot 4: Classical seasonal decomposition
    weekly_total = daily_total['sales'].resample('W').sum()
    decomposition = seasonal_decompose(weekly_total, model='multiplicative', period=52)
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    components = {
        'Observed': decomposition.observed,
        'Trend': decomposition.trend,
        'Seasonality': decomposition.seasonal,
        'Residual': decomposition.resid
    }
    colors = ['#2E86AB', '#E84855', '#F4A261', '#2A9D8F']
    for ax, (name, data), color in zip(axes, components.items(), colors):
        ax.plot(data, color=color, linewidth=1)
        ax.set_ylabel(name, fontweight='bold')
        ax.grid(True, alpha=0.3)
    axes[0].set_title('Seasonal Decomposition (Weekly, Multiplicative)', fontweight='bold')
    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    plt.savefig('outputs/plots/04_decomposition.png', dpi=150)
    plt.close()
    
    # Plot 5: Store 1, Item 1 deep dive
    s1i1 = df[(df['store'] == 1) & (df['item'] == 1)].set_index('date').sort_index()
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    axes[0].plot(s1i1.index, s1i1['sales'], color='#2E86AB', linewidth=0.8)
    axes[0].set_title('Store 1, Item 1 — Daily Sales', fontweight='bold')
    axes[0].set_ylabel('Units Sold')
    
    rolling_mean = s1i1['sales'].rolling(30, center=True).mean()
    axes[1].plot(s1i1.index, s1i1['sales'], color='#cccccc', linewidth=0.5, label='Daily')
    axes[1].plot(s1i1.index, rolling_mean, color='#E84855', linewidth=2, label='30-day Rolling Mean')
    axes[1].set_title('Store 1, Item 1 — With 30-Day Rolling Mean', fontweight='bold')
    axes[1].set_ylabel('Units Sold')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('outputs/plots/05_s1i1_deep_dive.png', dpi=150)
    plt.close()
    
    # 3. Feature Engineering
    df_features = generate_all_features(df)
    df_features.to_csv('data/processed/features.csv', index=False)
    logger.info("Engineered features saved to data/processed/features.csv")
    
    # 4. Model Training
    TARGET = 'sales'
    ID_COLS = ['date', 'store', 'item']
    FEATURE_COLS = [c for c in df_features.columns if c not in ID_COLS + [TARGET]]
    
    # Time-aware split: train before July 2017, validate on July-Dec 2017
    CUTOFF_DATE = '2017-07-01'
    train = df_features[df_features['date'] < CUTOFF_DATE].copy()
    test  = df_features[df_features['date'] >= CUTOFF_DATE].copy()
    
    X_train, y_train = train[FEATURE_COLS], train[TARGET]
    X_test, y_test   = test[FEATURE_COLS], test[TARGET]
    
    # Naive baseline evaluation (predicting last week's same day sales)
    baseline_preds = test['lag_7'].values
    baseline_mape, baseline_rmse = calculate_metrics(y_test, baseline_preds)
    
    # LightGBM training
    params = get_default_params()
    model, best_iter, residuals = train_lgb_model(X_train, y_train, X_test, y_test, params=params)
    
    # Generate predictions
    lgb_preds = predict_lgb(model, X_test, best_iter)
    lgb_mape, lgb_rmse = calculate_metrics(y_test, lgb_preds)
    
    # Log point evaluation metrics
    logger.info(f"Point Evaluation Complete.")
    logger.info(f"Baseline (Seasonal Naive) -> MAPE: {baseline_mape:.2f}% | RMSE: {baseline_rmse:.2f}")
    logger.info(f"Model (LightGBM)            -> MAPE: {lgb_mape:.2f}% | RMSE: {lgb_rmse:.2f}")
    
    # Save predictions
    results_df = test[ID_COLS + [TARGET]].copy()
    results_df['baseline_pred'] = baseline_preds
    results_df['model_pred'] = lgb_preds
    results_df.to_csv('outputs/results/test_predictions.csv', index=False)
    logger.info("Test predictions exported to outputs/results/test_predictions.csv")
    
    # 5. Feature Importance Visualisation
    importance_df = pd.DataFrame({
        'feature': model.feature_name(),
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False).head(20)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors_fi = ['#E84855' if i < 5 else '#2E86AB' for i in range(len(importance_df))]
    ax.barh(importance_df['feature'][::-1], importance_df['importance'][::-1],
            color=colors_fi[::-1], edgecolor='black', linewidth=0.3)
    ax.set_title('Top 20 Feature Importances (LightGBM — Gain)', fontweight='bold')
    ax.set_xlabel('Feature Importance (Gain)')
    ax.axvline(0, color='black', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('outputs/plots/06_feature_importance.png', dpi=150)
    plt.close()
    
    # Plot 7: Actual vs Predicted
    s, i = 1, 1
    mask = (test['store'] == s) & (test['item'] == i)
    test_sub = test[mask]
    actual_sub = y_test[mask].values
    predicted_sub = lgb_preds[mask.values]
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    axes[0].plot(test_sub['date'], actual_sub, label='Actual', color='#2E86AB', linewidth=1.2)
    axes[0].plot(test_sub['date'], predicted_sub, label='LightGBM Forecast',
                 color='#E84855', linewidth=1.2, linestyle='--')
    axes[0].set_title(f'Store {s}, Item {i} — Actual vs Forecast (Full Test Period)', fontweight='bold')
    axes[0].set_ylabel('Daily Sales')
    axes[0].legend()
    
    axes[1].plot(test_sub['date'][:60], actual_sub[:60], label='Actual', color='#2E86AB', linewidth=1.5, marker='o', markersize=3)
    axes[1].plot(test_sub['date'][:60], predicted_sub[:60], label='LightGBM Forecast',
                 color='#E84855', linewidth=1.5, linestyle='--', marker='s', markersize=3)
    axes[1].set_title(f'Store {s}, Item {i} — First 60 Days (Zoomed)', fontweight='bold')
    axes[1].set_ylabel('Daily Sales')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('outputs/plots/07_actual_vs_predicted.png', dpi=150)
    plt.close()
    
    # Plot 8: Residual Analysis
    residuals_test = y_test.values - lgb_preds
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].scatter(range(len(residuals_test)), residuals_test, alpha=0.3, s=1, color='#2E86AB')
    axes[0].axhline(0, color='red', linewidth=1)
    axes[0].set_title('Residuals Over Time', fontweight='bold')
    axes[0].set_xlabel('Observation Index')
    axes[0].set_ylabel('Residual')
    
    axes[1].hist(residuals_test, bins=80, color='#2E86AB', edgecolor='white', linewidth=0.3)
    axes[1].axvline(0, color='red', linewidth=1.5)
    axes[1].set_title('Residual Distribution', fontweight='bold')
    axes[1].set_xlabel('Residual Value')
    
    mean_res = np.mean(residuals_test)
    std_res = np.std(residuals_test)
    axes[2].text(0.5, 0.5,
                 f'Mean: {mean_res:.2f}\nStd: {std_res:.2f}\nMin: {residuals_test.min():.0f}\nMax: {residuals_test.max():.0f}',
                 transform=axes[2].transAxes, fontsize=14,
                 verticalalignment='center', horizontalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[2].set_title('Residual Statistics', fontweight='bold')
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig('outputs/plots/08_residual_analysis.png', dpi=150)
    plt.close()
    
    # 6. Walk-Forward Backtesting
    cutoff_dates = ['2016-07-01', '2016-10-01', '2017-01-01', '2017-04-01']
    backtest_results = run_backtest(df_features, FEATURE_COLS, TARGET, cutoff_dates, params)
    backtest_results.to_csv('outputs/results/backtest_results.csv', index=False)
    logger.info("Backtest walk-forward simulation complete. Logs exported.")
    
    # 7. Inventory Cost Impact Translation
    UNIT_COST = 150
    SELLING_PRICE = 250
    
    baseline_impact = calculate_inventory_impact(y_test.values, baseline_preds, UNIT_COST, SELLING_PRICE)
    model_impact = calculate_inventory_impact(y_test.values, lgb_preds, UNIT_COST, SELLING_PRICE)
    cost_saved = baseline_impact['total_cost'] - model_impact['total_cost']
    
    logger.info("--- Inventory Cost Translation ---")
    logger.info(f"Baseline (Seasonal Naive) Total Error Cost: INR {baseline_impact['total_cost']:,.2f}")
    logger.info(f"Model (LightGBM) Total Error Cost:            INR {model_impact['total_cost']:,.2f}")
    logger.info(f"Financial Value Added (Net Cost Savings):    INR {cost_saved:,.2f} ({cost_saved/baseline_impact['total_cost']*100:.1f}% reduction)")
    
    metrics_summary = pd.DataFrame({
        'Metric': ['Understocked Units', 'Overstocked Units', 'Stockout Cost', 'Overstock Cost', 'Total Inventory Cost'],
        'Baseline (Seasonal Naive)': [
            baseline_impact['total_units_understocked'],
            baseline_impact['total_units_overstocked'],
            baseline_impact['stockout_cost'],
            baseline_impact['overstock_cost'],
            baseline_impact['total_cost']
        ],
        'LightGBM': [
            model_impact['total_units_understocked'],
            model_impact['total_units_overstocked'],
            model_impact['stockout_cost'],
            model_impact['overstock_cost'],
            model_impact['total_cost']
        ]
    })
    metrics_summary.to_csv('outputs/results/inventory_impact_summary.csv', index=False)
    
    # Plot 9: Business Impact comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    categories = ['Stockout\nCost', 'Overstock\nCost', 'Total\nCost']
    baseline_vals = [baseline_impact['stockout_cost'], baseline_impact['overstock_cost'], baseline_impact['total_cost']]
    model_vals = [model_impact['stockout_cost'], model_impact['overstock_cost'], model_impact['total_cost']]
    
    x = np.arange(len(categories))
    width = 0.35
    bars1 = axes[0].bar(x - width/2, baseline_vals, width, label='Seasonal Naive', color='#E84855', alpha=0.8, edgecolor='black')
    bars2 = axes[0].bar(x + width/2, model_vals, width, label='LightGBM', color='#2E86AB', alpha=0.8, edgecolor='black')
    axes[0].set_title('Inventory Cost: Baseline vs Model', fontweight='bold')
    axes[0].set_ylabel('Cost (₹)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].legend()
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x:,.0f}'))
    
    for bar in bars1:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                     f'₹{bar.get_height():,.0f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                     f'₹{bar.get_height():,.0f}', ha='center', va='bottom', fontsize=8)
                     
    savings_pct = cost_saved / baseline_impact['total_cost'] * 100
    axes[1].pie(
        [model_impact['total_cost'], cost_saved],
        labels=['Remaining Cost\n(with LightGBM)', f'Cost Saved\n₹{cost_saved:,.0f}\n({savings_pct:.1f}%)'],
        colors=['#2E86AB', '#2A9D8F'],
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'edgecolor': 'black', 'linewidth': 1}
    )
    axes[1].set_title('Proportion of Costs Eliminated by Model', fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/plots/09_business_impact.png', dpi=150)
    plt.close()
    
    # 8. Prediction Intervals
    logger.info("Generating bootstrap prediction intervals...")
    lower_bounds, upper_bounds, point_forecasts = bootstrap_prediction_intervals(
        model, X_test, residuals, best_iter
    )
    
    coverage = np.mean((y_test.values >= lower_bounds) & (y_test.values <= upper_bounds)) * 100
    logger.info(f"90% Prediction Interval Coverage: {coverage:.2f}% (Target: ~90%)")
    
    pi_df = test[ID_COLS + [TARGET]].copy()
    pi_df['forecast_lower'] = lower_bounds
    pi_df['forecast_point'] = point_forecasts
    pi_df['forecast_upper'] = upper_bounds
    pi_df.to_csv('outputs/results/prediction_intervals.csv', index=False)
    
    # Plot 10: Store 1, Item 1 with intervals
    idx = mask.values
    fig, ax = plt.subplots(figsize=(16, 6))
    dates_sub = test[mask]['date']
    ax.fill_between(dates_sub, lower_bounds[idx], upper_bounds[idx],
                    alpha=0.2, color='#2E86AB', label='90% Prediction Interval')
    ax.plot(dates_sub, point_forecasts[idx], color='#E84855', linewidth=1.5,
            linestyle='--', label='Point Forecast')
    ax.plot(dates_sub, y_test.values[idx], color='#2E86AB', linewidth=1.2,
            label='Actual Sales')
    ax.set_title(f'Store {s}, Item {i} — Forecast with Prediction Intervals', fontweight='bold')
    ax.set_ylabel('Daily Sales')
    ax.legend()
    plt.tight_layout()
    plt.savefig('outputs/plots/10_prediction_intervals.png', dpi=150)
    plt.close()
    
    logger.info("Pipeline execution complete. All outputs generated successfully.")

if __name__ == '__main__':
    main()
