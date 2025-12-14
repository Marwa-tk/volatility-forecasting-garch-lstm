
"""
evaluation_comparison.py
------------------------
Comparaison des modèles GARCH et LSTM pour la prévision de la volatilité
Auteur : En-Nasry Salma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy import stats


def calcul_mse_rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mse, rmse


def log_vraisemblance(erreur):
    return -0.5 * np.sum(np.log(2 * np.pi) + erreur**2)


def diebold_mariano(e1, e2):
    d = e1**2 - e2**2
    dm_stat = np.mean(d) / np.sqrt(np.var(d, ddof=1) / len(d))
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value


def charger_donnees(real_path, garch_path, lstm_path):
    real = pd.read_csv(real_path)
    garch = pd.read_csv(garch_path)
    lstm = pd.read_csv(lstm_path)

    real['date'] = pd.to_datetime(real['date'])
    garch['date'] = pd.to_datetime(garch['date'])
    lstm['date'] = pd.to_datetime(lstm['date'])

    df = real[['date', 'rv20']]         .merge(garch[['date', 'volatility_garch']], on='date')         .merge(lstm[['date', 'volatility']], on='date')

    df.columns = ['date', 'vol_real', 'vol_garch', 'vol_lstm']
    return df


def evaluer_modele(df, nom_actif):
    err_garch = df['vol_real'] - df['vol_garch']
    err_lstm = df['vol_real'] - df['vol_lstm']

    mse_garch, rmse_garch = calcul_mse_rmse(df['vol_real'], df['vol_garch'])
    mse_lstm, rmse_lstm = calcul_mse_rmse(df['vol_real'], df['vol_lstm'])

    ll_garch = log_vraisemblance(err_garch)
    ll_lstm = log_vraisemblance(err_lstm)

    dm_stat, p_value = diebold_mariano(err_garch, err_lstm)

    print(f"
=== {nom_actif} ===")
    print(f"GARCH | MSE: {mse_garch:.6f} | RMSE: {rmse_garch:.6f} | LL: {ll_garch:.2f}")
    print(f"LSTM  | MSE: {mse_lstm:.6f} | RMSE: {rmse_lstm:.6f} | LL: {ll_lstm:.2f}")
    print(f"DM test | Stat: {dm_stat:.4f} | p-value: {p_value:.4f}")


def plot_volatilite(df, titre):
    plt.figure(figsize=(12,5))
    plt.plot(df['date'], df['vol_real'], label='Volatilité réalisée', linewidth=2)
    plt.plot(df['date'], df['vol_garch'], label='GARCH', linestyle='--')
    plt.plot(df['date'], df['vol_lstm'], label='LSTM', linestyle=':')
    plt.title(titre)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    gspc = charger_donnees(
        "GSPC_sample.csv",
        "GSPC_volatility_garch.csv",
        "GSPC_volatility_lstm.csv"
    )

    evaluer_modele(gspc, "S&P 500 (GSPC)")
    plot_volatilite(gspc, "Prévision de la volatilité — S&P 500")


    btc = charger_donnees(
        "BTC-USD_sample.csv",
        "BTCUSD_volatility_garch.csv",
        "BTCUSD_volatility_lstm.csv"
    )

    evaluer_modele(btc, "Bitcoin (BTC-USD)")
    plot_volatilite(btc, "Prévision de la volatilité — Bitcoin")
