import os
from typing import Union

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import cross_val_score


def exp_weights(window, span):
    return np.exp(np.linspace(0, -window, window) / span)

# Custom function to apply with rolling
def custom_ewm(series, span):
    weights = exp_weights(len(series), span)
    return np.average(series, weights=weights)

def calculate_la_stat_ewm(
    df: pd.DataFrame,
    col: str,
    window: int,
    span: Union[int, float],
    prior_cc: Union[int, float] = 6,
    periods = 1,
    # max_periods = 10000,
) -> pd.DataFrame:
    hdf = df.copy()
    # compute diff from previous time
    hdf[col] = hdf[col].diff(periods=periods).fillna(0)

    # within season values
    day_ewm = hdf.groupby("DAY")[col].transform(lambda x: x.rolling(window=int(window), min_periods=1).apply(lambda y: custom_ewm(y, span), raw=True))
    day_ewm = (
        day_ewm.groupby([hdf["DAY"], hdf["timestamp"]])
        .last()
        .groupby("DAY")
        .shift()
        .reset_index(drop=False)
    )

    # cross season values
    ewm = hdf[col].transform(lambda x: x.rolling(window=int(window), min_periods=1).apply(lambda y: custom_ewm(y, span), raw=True))
    ewm = ewm.shift().ffill().reset_index(drop=False)

    day_cc = day_ewm.groupby("DAY").cumcount()

    # use cross season as prior
    ewm[col] = (
        (day_ewm[col] * day_cc + ewm[col] * prior_cc) / (day_cc + prior_cc)
    ).fillna(ewm[col])

    ewm.columns = ["Index", f"EWM_AVG_{col}"]

    ewm = ewm[[f"EWM_AVG_{col}"]]
    ewm = ewm.ffill().bfill()
    return ewm

def compute_price(df, window, span, prior_cc):
    # this df should contain the price history for Orchids, sunlight, and humidity
    ewm_frames = []
    for period in range(3):
        ewm_period = []
        for var in ["SUNLIGHT", "HUMIDITY", "ORCHIDS"]:
            ewm_stat = calculate_la_stat_ewm(df, var, window, span, prior_cc, period + 1)
            ewm_period.append(ewm_stat)
        ewm_combined = pd.concat(ewm_period, axis=1)
        ewm_frames.append(ewm_combined)
    
    COEFS = [0.04071376, 0.00244406, -0.00799631]
    INTERCEPT = 0.0
    ewm = pd.concat(ewm_frames, axis=1).iloc[-1]
    all_neg = (ewm < 0).all()*  0.04292393
    all_pos = (ewm > 0).all() * 0.0
    return np.dot(ewm, COEFS) + all_neg + all_pos + INTERCEPT

def compute_regression(params, df, lag, response="ORCHIDS"):
    # Group parameters in tuples of (span, prior_cc) for each variable across three periods
    param_groups = {
        "SUNLIGHT": params[0:lag*3],
        "HUMIDITY": params[lag*3:lag*6],
        "ORCHIDS": params[lag*6:lag*9]
    }

    ewm_frames = []

    for period in range(lag):
        period_index = period * 2
        ewm_period = []
        for var, group in param_groups.items():
            window = group[period_index]
            span = group[period_index + 1]
            prior_cc = group[period_index + 2]
            ewm_stat = calculate_la_stat_ewm(df, var, window, span, prior_cc, period + 1)
            ewm_period.append(ewm_stat)
        ewm_combined = pd.concat(ewm_period, axis=1)
        ewm_frames.append(ewm_combined)

    ewm = pd.concat(ewm_frames, axis=1).iloc[1:]
    # all negative
    ewm["all_neg"] = (ewm < 0).all(axis=1).astype(int)
    ewm["all_pos"] = (ewm > 0).all(axis=1).astype(int)

    response_data = df[response].diff().iloc[1:]
    
    # ridge = RidgeCV()
    # ridge.fit(ewm, response_data)
    
    # best_alpha = ridge.alpha_
    ridge_final = Ridge(alpha=1000)
    # [0.03925727 0.00445401 0.00366386 0.07510591 0.        ]
    ridge_final.fit(ewm, response_data)
    
    mse_scores = -cross_val_score(ridge_final, ewm, response_data, scoring='neg_mean_squared_error', cv=5)
    return np.mean(mse_scores)

def optimize_params(df, lag = 1):
    df['no_sun'] = (df.SUNLIGHT<2500).astype(int)
    df['cum_sum'] = df.groupby('DAY')['no_sun'].transform('cumsum') 
    df['sun_greater_7_hrs'] = df.groupby('DAY')['cum_sum'].transform(lambda x: (x>5833).cumsum())
# [ 669.92646802 1977.98371139  926.31695217  493.57537655 4405.02947201
#   537.73672044   88.27088439  567.85520167  533.24102539]
    bounds = [(1, 1000), (1, 5000), (1, 1000)] * 3*lag  # Three times for each of the three variables SUNLIGHT, HUMIDITY, ORCHIDS
    result = differential_evolution(
        compute_regression,
        bounds,
        args=(df,lag, ),
        tol=1e-5,
        disp=True,
        workers=os.cpu_count() - 2
    )
    return result.x


def load_concat_data():
    # Load the data
    df = None
    for day in [-1, 0, 1]:
        df1 = pd.read_csv(f"data/round-2-island-data-bottle/prices_round_2_day_{day}.csv", sep=";")

        # Concatenate the data
        df = pd.concat([df, df1])
    return df


if __name__ == "__main__":
    # Test the function
    df = load_concat_data()
    optimize_params(df)
    print("done")