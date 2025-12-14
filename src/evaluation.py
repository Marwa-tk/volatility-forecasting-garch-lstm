"""
Evaluation des modèles de volatilité
GARCH vs LSTM

Auteur : Salma EN-NASRY
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return rmse, mae, mape


def diebold_mariano(e1, e2):
    d = e1**2 - e2**2
    dm_stat = np.mean(d) / np.sqrt(np.var(d) / len(d))
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    return dm_stat, p_value


def evaluate(ticker):
    actual = pd.read_csv(f"data/processed/{ticker}.csv")
    garch = pd.read_csv(f"data/predictions/garch_{ticker}.csv")
    lstm = pd.read_csv(f"data/predictions/lstm_{ticker}.csv")

    y = actual["rv20"]
    g = garch["volatility_garch"]
    l = lstm["volatility_lstm"]

    rmse_g, mae_g, mape_g = compute_metrics(y, g)
    rmse_l, mae_l, mape_l = compute_metrics(y, l)

    dm, p = diebold_mariano(y - g, y - l)

    return {
        "Ticker": ticker,
        "RMSE_GARCH": rmse_g,
        "RMSE_LSTM": rmse_l,
        "MAE_GARCH": mae_g,
        "MAE_LSTM": mae_l,
        "MAPE_GARCH": mape_g,
        "MAPE_LSTM": mape_l,
        "DM_stat": dm,
        "p_value": p
    }


if __name__ == "__main__":
    results = []
    for tk in ["GSPC", "BTC-USD"]:
        results.append(evaluate(tk))

    df = pd.DataFrame(results)
    df.to_csv("results/evaluation_summary.csv", index=False)
    print(df)

