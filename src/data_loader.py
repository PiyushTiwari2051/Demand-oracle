import os
import logging
import subprocess
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def load_data(filepath: str = 'data/raw/train.csv', force_generate: bool = False) -> pd.DataFrame:
    """
    Loads raw store-item sales dataset. Falls back to generating a realistic
    synthetic dataset if Kaggle CLI fails or if raw data is missing.
    """
    if force_generate or not os.path.exists(filepath):
        logger.info(f"Dataset not found at {filepath}. Attempting to resolve...")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if not force_generate:
            try:
                logger.info("Attempting download from Kaggle...")
                # Kaggle CLI requires credentials to be configured in ~/.kaggle/
                cmd = "kaggle competitions download -c demand-forecasting-kernels-only"
                subprocess.run(cmd, shell=True, check=True)
                
                import zipfile
                zip_path = "demand-forecasting-kernels-only.zip"
                if os.path.exists(zip_path):
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(os.path.dirname(filepath))
                    os.remove(zip_path)
                    logger.info("Successfully fetched Kaggle dataset.")
            except Exception as e:
                logger.warning(f"Kaggle download failed: {e}. Falling back to synthetic generator.")
                generate_synthetic_data(filepath)
        else:
            generate_synthetic_data(filepath)
            
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    validate_dataset(df)
    return df

def generate_synthetic_data(filepath: str):
    """
    Generates a realistic synthetic sales dataset mirroring Kaggle's schema:
    913,000 rows (10 stores * 50 items * 1826 days) with seasonality & trend.
    """
    logger.info("Generating synthetic sales data...")
    dates = pd.date_range(start='2013-01-01', end='2017-12-31', freq='D')
    num_days = len(dates)
    num_stores, num_items = 10, 50
    
    np.random.seed(42)
    # Varying scale factors to mimic store volume and item popularity
    store_multipliers = np.random.uniform(0.7, 1.5, size=num_stores)
    item_multipliers = np.clip(np.random.exponential(scale=1.0, size=num_items), 0.2, 5.0)
    
    store_ids = np.arange(1, num_stores + 1)
    item_ids = np.arange(1, num_items + 1)
    
    # 5-year global upward trend
    trend_signal = np.linspace(20, 35, num_days)
    
    # Weekly seasonality: weekend spikes (Fri, Sat, Sun)
    weekly_seasonality = np.array([1.0, 1.02, 1.05, 1.1, 1.25, 1.4, 1.3])
    weekly_signal = weekly_seasonality[dates.dayofweek]
    
    # Yearly seasonality: peak in July, dip in January
    yearly_signal = 1.0 + 0.3 * np.sin(2 * np.pi * (dates.dayofyear - 80) / 365)
    
    daily_base = trend_signal * weekly_signal * yearly_signal
    
    dfs = []
    for store in store_ids:
        s_mult = store_multipliers[store - 1]
        for item in item_ids:
            i_mult = item_multipliers[item - 1]
            
            expected_sales = daily_base * s_mult * i_mult
            # Add heteroscedastic noise (variance scale with sales volume)
            noise = np.random.normal(0, np.sqrt(expected_sales) * 0.5, size=num_days)
            sales = expected_sales + noise
            
            # NOTE: Clip at 1 to prevent division by zero / division by epsilon in MAPE
            sales = np.maximum(np.round(sales), 1).astype(int)
            
            dfs.append(pd.DataFrame({
                'date': dates,
                'store': store,
                'item': item,
                'sales': sales
            }))
            
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(['store', 'item', 'date']).reset_index(drop=True)
    df.to_csv(filepath, index=False)
    logger.info(f"Synthetic dataset saved to {filepath}. Shape: {df.shape}")

def validate_dataset(df: pd.DataFrame):
    """Run sanity checks to guarantee dataset integrity."""
    required = {'date', 'store', 'item', 'sales'}
    missing = required - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
    assert not df.isnull().any().any(), "Dataset contains nulls!"
    assert df['store'].nunique() == 10, f"Expected 10 stores, got {df['store'].nunique()}"
    assert df['item'].nunique() == 50, f"Expected 50 items, got {df['item'].nunique()}"
    assert len(df) == 913000, f"Expected 913,000 rows, got {len(df)}"
