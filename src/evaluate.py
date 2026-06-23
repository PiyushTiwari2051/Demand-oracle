import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import lightgbm as lgb
from src.model import predict_lgb

logger = logging.getLogger(__name__)

def calculate_metrics(y_true, y_pred) -> tuple:
    """Calculate MAPE and RMSE."""
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mape, rmse

def run_backtest(df: pd.DataFrame, feature_cols: list, target: str, 
                 cutoff_dates: list, model_params: dict, predict_horizon: int = 90) -> pd.DataFrame:
    """
    Executes walk-forward backtesting (historical simulation) over multiple cutoff points.
    Trains on historical data and evaluates on a fixed forward horizon.
    """
    results = []
    
    for cutoff in cutoff_dates:
        logger.info(f"Running backtest fold for cutoff: {cutoff}")
        
        train_bt = df[df['date'] < cutoff]
        cutoff_dt = pd.to_datetime(cutoff)
        test_end = cutoff_dt + pd.Timedelta(days=predict_horizon)
        test_bt = df[(df['date'] >= cutoff) & (df['date'] < test_end)]
        
        if len(test_bt) == 0:
            logger.warning(f"Skipping fold {cutoff}: target evaluation window has no observations.")
            continue
            
        X_tr, y_tr = train_bt[feature_cols], train_bt[target]
        X_te, y_te = test_bt[feature_cols], test_bt[target]
        
        tr_data = lgb.Dataset(X_tr, label=y_tr)
        te_data = lgb.Dataset(X_te, label=y_te, reference=tr_data)
        
        # Train fold model
        model = lgb.train(
            model_params,
            tr_data,
            num_boost_round=500,
            valid_sets=[te_data],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=0) # Suppress evaluation spam
            ]
        )
        
        # Generate model point forecasts
        preds = predict_lgb(model, X_te, model.best_iteration)
        
        # Baseline: Seasonal Naive (same day last week)
        baseline = test_bt['lag_7'].values
        
        mape_model, rmse_model = calculate_metrics(y_te, preds)
        mape_baseline, rmse_baseline = calculate_metrics(y_te, baseline)
        
        improvement_pct = (mape_baseline - mape_model) / mape_baseline * 100
        
        results.append({
            'cutoff': cutoff,
            'test_days': len(test_bt['date'].unique()),
            'mape_model': round(mape_model, 3),
            'mape_baseline': round(mape_baseline, 3),
            'rmse_model': round(rmse_model, 3),
            'rmse_baseline': round(rmse_baseline, 3),
            'improvement_pct': round(improvement_pct, 1)
        })
        
        logger.info(f"Fold {cutoff} complete. Model MAPE: {mape_model:.2f}% | Baseline: {mape_baseline:.2f}% | Improvement: {improvement_pct:.1f}%")
        
    return pd.DataFrame(results)
