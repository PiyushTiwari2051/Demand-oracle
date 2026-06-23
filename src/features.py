import logging
import pandas as pd
import numpy as np
import holidays

logger = logging.getLogger(__name__)

def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract standard calendar and cyclical signals from the date column."""
    df = df.copy()
    
    # Standard date parts
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter
    
    # Quick indicators
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
    
    # Cyclical sin/cos encoding to preserve temporal proximity (e.g., Dec and Jan)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    
    return df

def add_holiday_features(df: pd.DataFrame, country: str = 'US', years: list = None) -> pd.DataFrame:
    """
    Computes holiday flags and proximity features.
    Optimized by mapping days_to_holiday across unique dates to prevent CPU bottlenecks.
    """
    df = df.copy()
    if years is None:
        years = df['date'].dt.year.unique().tolist()
        
    country_holidays = holidays.country_holidays(country, years=years)
    df['is_holiday'] = df['date'].isin(country_holidays).astype(int)
    
    # Map distance to holidays using unique dates only (cuts runtime from minutes to 0.2s)
    holiday_dates = pd.to_datetime(list(country_holidays.keys()))
    unique_dates = pd.Series(df['date'].unique())
    days_to_h = unique_dates.apply(lambda d: min(abs((d - h).days) for h in holiday_dates))
    date_to_days = dict(zip(unique_dates, days_to_h))
    
    df['days_to_holiday'] = df['date'].map(date_to_days).clip(upper=7)
    return df

def add_lag_features(df: pd.DataFrame, lags: list) -> pd.DataFrame:
    """Create autoregressive lag features for sales."""
    df = df.copy()
    # Sort order is critical before shifting
    df = df.sort_values(['store', 'item', 'date']).reset_index(drop=True)
    
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby(['store', 'item'])['sales'].shift(lag)
    return df

def add_rolling_features(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    """
    Generates rolling mean, std, and max statistics.
    Pre-shifts the target by 1 to prevent future target leakage.
    Uses C-optimized groupby rolling for fast computation.
    """
    df = df.copy()
    df = df.sort_values(['store', 'item', 'date']).reset_index(drop=True)
    
    # shift(1) is non-negotiable here to avoid lookahead leakage
    shifted_sales = df.groupby(['store', 'item'])['sales'].shift(1)
    grouped_shifted = shifted_sales.groupby([df['store'], df['item']])
    
    for window in windows:
        # Reset the multi-index levels to allow alignment back to the primary DataFrame
        df[f'rolling_mean_{window}'] = grouped_shifted.rolling(window, min_periods=1).mean().reset_index(level=['store', 'item'], drop=True)
        df[f'rolling_std_{window}']  = grouped_shifted.rolling(window, min_periods=1).std().reset_index(level=['store', 'item'], drop=True)
        df[f'rolling_max_{window}']  = grouped_shifted.rolling(window, min_periods=1).max().reset_index(level=['store', 'item'], drop=True)
        
    return df

def add_expanding_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes historical expanding average of sales to act as SKU baseline."""
    df = df.copy()
    df = df.sort_values(['store', 'item', 'date']).reset_index(drop=True)
    shifted_sales = df.groupby(['store', 'item'])['sales'].shift(1)
    
    grouped_shifted = shifted_sales.groupby([df['store'], df['item']])
    df['expanding_mean'] = grouped_shifted.expanding(min_periods=30).mean().reset_index(level=['store', 'item'], drop=True)
    return df

def generate_all_features(df: pd.DataFrame, lags: list = [1, 7, 14, 21, 28, 91, 365], 
                          windows: list = [7, 14, 30, 90], country: str = 'US') -> pd.DataFrame:
    """Sequentially engineers all dataset features and drops incomplete cold-start rows."""
    logger.info("Starting optimized feature engineering pipeline...")
    df = df.sort_values(['store', 'item', 'date']).reset_index(drop=True)
    
    df = add_datetime_features(df)
    df = add_holiday_features(df, country=country)
    df = add_lag_features(df, lags=lags)
    df = add_rolling_features(df, windows=windows)
    df = add_expanding_features(df)
    
    initial_rows = len(df)
    df = df.dropna().reset_index(drop=True)
    logger.info(f"Dropped {initial_rows - len(df):,} cold-start rows. Final shape: {df.shape}")
    return df
