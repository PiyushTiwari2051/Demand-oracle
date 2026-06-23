import logging
import numpy as np
import lightgbm as lgb

logger = logging.getLogger(__name__)

def get_default_params() -> dict:
    """Standard, production-tested hyperparameters for LightGBM regression."""
    return {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 127,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'verbose': -1
    }

def train_lgb_model(X_train, y_train, X_val, y_val, params: dict = None, 
                    num_boost_round: int = 1000, early_stopping_rounds: int = 50):
    """
    Trains a LightGBM regressor with validation early stopping.
    Calculates residuals on the training set for bootstrapping prediction intervals.
    """
    if params is None:
        params = get_default_params()
        
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    logger.info("Training LightGBM model...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )
    
    best_iteration = model.best_iteration
    logger.info(f"Model training finished. Best iteration: {best_iteration}")
    
    # Pre-calculate training residuals to represent model error distribution
    train_preds = model.predict(X_train, num_iteration=best_iteration)
    train_residuals = y_train.values - train_preds
    
    return model, best_iteration, train_residuals

def predict_lgb(model, X, best_iteration: int) -> np.ndarray:
    """Generate point forecasts clipped at 0 (since sales cannot be negative)."""
    preds = model.predict(X, num_iteration=best_iteration)
    return np.maximum(preds, 0)

def bootstrap_prediction_intervals(model, X, train_residuals: np.ndarray, 
                                   best_iteration: int, n_bootstrap: int = 100, 
                                   confidence: float = 0.9) -> tuple:
    """
    Estimates prediction intervals using non-parametric Residual Bootstrap.
    Adds randomly sampled historical residuals directly to the point forecast.
    """
    point_forecast = predict_lgb(model, X, best_iteration)
    bootstrap_preds = np.zeros((n_bootstrap, len(X)))
    
    for b in range(n_bootstrap):
        # Sample residuals with replacement and add to point forecast
        sampled_residuals = np.random.choice(train_residuals, size=len(X), replace=True)
        bootstrap_preds[b] = point_forecast + sampled_residuals
        
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_preds, (alpha / 2) * 100, axis=0)
    upper = np.percentile(bootstrap_preds, (1 - alpha / 2) * 100, axis=0)
    
    # Clip lower bound at 0 since negative sales are impossible
    lower = np.maximum(lower, 0)
    
    return lower, upper, point_forecast
