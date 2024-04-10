import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import cross_val_score


def load_data(product: str, price_data: bool) -> pd.DataFrame:
    if price_data:
        data_2 = pd.read_csv('data/prices_round_1_day_-2.csv', sep=';')
        data_1 = pd.read_csv('data/prices_round_1_day_-1.csv', sep=';')
        data_0 = pd.read_csv('data/prices_round_1_day_0.csv', sep=';')
    else:
        data_2 = pd.read_csv('data/trades_round_1_day_-2_nn.csv', sep=';')
        data_1 = pd.read_csv('data/trades_round_1_day_-1_nn.csv', sep=';')
        data_0 = pd.read_csv('data/trades_round_1_day_0_nn.csv', sep=';')

    data = pd.concat([data_2, data_1, data_0], axis=0).reset_index(drop=True)
    data = data[data['product'] == product]
    return data

def lin_reg(data: pd.DataFrame, num_timesteps: int) -> Tuple[float, np.ndarray]:
    data.sort_values(['day', 'timestamp'], inplace=True)
    for i in range(1, num_timesteps + 1):
        data[f'price_t-{i}'] = data['mid_price'].shift(i)
    features = [f'price_t-{i}' for i in range(1, num_timesteps + 1)]
    data = data.loc[data[features].isna().sum(axis=1) == 0]
    X = data[features]
    y = data['mid_price']
    
    # Initialize RidgeCV to find the best alpha
    ridge_cv = RidgeCV(alphas=np.logspace(-6, 6, 13), scoring='neg_mean_squared_error', cv=5)
    
    # Fit RidgeCV on the data to perform cross-validation and find the best alpha
    ridge_cv.fit(X, y)
    
    # refit on all the data using the best alpha
    best_alpha = ridge_cv.alpha_
    ridge_final = Ridge(alpha=best_alpha)
    ridge_final.fit(X, y)
    
    # get the mean mse score from cross-validation
    mse_scores = -cross_val_score(ridge_final, X, y, scoring='neg_mean_squared_error', cv=5)
    mean_mse = np.mean(mse_scores)
    
    # Get the coefficients from the final model
    coefs = ridge_final.coef_
    
    return mean_mse, coefs

def compare_mse(prices: pd.DataFrame, time_list: List[int]) -> None:
    best_mse = np.inf
    best_time = 0
    best_coefs = None
    mean_mses = []
    for time in time_list:
        mean_mse, coefs = lin_reg(prices, time)
        mean_mses.append(mean_mse)
        if mean_mse < best_mse:
            best_mse = mean_mse
            best_time = time
            best_coefs = coefs

    mean_mses = pd.DataFrame(mean_mses, columns=['cv_score'])
    print(mean_mses)
    mean_mses['time'] = time_list
    print(f"Best MSE: {best_mse}")
    print(f"Best time: {best_time}")
    print(f"Best coefs: {best_coefs}")
    return best_mse, best_time, best_coefs

if __name__ == '__main__':
    product = 'STARFRUIT'
    prices = load_data(product, price_data=True)
    compare_mse(prices, np.arange(1, 50, 1))
    # Best MSE: 1.9167838633948413
    # Best time: 15
    # Best coefs: [ 0.2920955   0.20671938  0.14077617  0.10025522  0.08580541  0.06038695
    #   0.03888277  0.00594952  0.02262225  0.01394354  0.0164973   0.00535559
    #   0.00513494  0.00572899 -0.00049075]