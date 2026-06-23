import numpy as np

def calculate_inventory_impact(actual: np.ndarray, predicted: np.ndarray, 
                               unit_cost: float, selling_price: float,
                               holding_cost_pct: float = 0.25, 
                               stockout_multiplier: float = 1.5) -> dict:
    """
    Translates mathematical forecast error into concrete financial cost:
    
    1. Under-forecasting (predicted < actual) causes stockouts:
       Cost = lost revenue * stockout multiplier (brand damage penalty).
       
    2. Over-forecasting (predicted > actual) causes overstocks:
       Cost = excess inventory * daily holding rate.
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    under_forecast = np.maximum(actual - predicted, 0)
    over_forecast  = np.maximum(predicted - actual, 0)
    
    # Cost of lost sales + brand reputation penalty
    stockout_cost_per_unit = selling_price * stockout_multiplier
    total_stockout_cost = np.sum(under_forecast) * stockout_cost_per_unit
    
    # Cost of holding capital tied up in warehouses
    holding_cost_per_unit = unit_cost * holding_cost_pct / 365
    total_overstock_cost = np.sum(over_forecast) * holding_cost_per_unit
    
    total_cost = total_stockout_cost + total_overstock_cost
    total_units_understocked = np.sum(under_forecast)
    total_units_overstocked  = np.sum(over_forecast)
    
    return {
        'total_units_understocked': int(total_units_understocked),
        'total_units_overstocked': int(total_units_overstocked),
        'stockout_cost': round(total_stockout_cost, 2),
        'overstock_cost': round(total_overstock_cost, 2),
        'total_cost': round(total_cost, 2),
        'avg_daily_understocked': round(total_units_understocked / len(actual), 2),
        'avg_daily_overstocked': round(total_units_overstocked / len(actual), 2),
    }
