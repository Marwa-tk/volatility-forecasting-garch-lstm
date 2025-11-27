#!/usr/bin/env bash
set -euo pipefail

# === Import des fonctions utilitaires ===
. "$(dirname "$0")/utils.sh"

yellow "▶ STEP_1 — Téléchargement et Prétraitement des données (S&P 500, BTC-USD)"
ensure_dir data/raw
ensure_dir data/processed
load_env

# === Paramètres par défaut (si .env absent) ===
TICKERS="${TICKERS:-^GSPC,BTC-USD}"
START_DATE="${START_DATE:-2018-01-01}"
END_DATE="${END_DATE:-2025-01-01}"
INTERVAL="${INTERVAL:-1d}"

yellow "Paramètres utilisés → TICKERS=$TICKERS | $START_DATE → $END_DATE | INTERVAL=$INTERVAL"

python - << 'PY'
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from statsmodels.tsa.stattools import adfuller, kpss


sns.set_style("whitegrid")

tickers = os.getenv("TICKERS", "^GSPC,BTC-USD").split(",")
start   = os.getenv("START_DATE", "2018-01-01")
end     = os.getenv("END_DATE", "2025-01-01")
interval= os.getenv("INTERVAL", "1d")


os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("figures", exist_ok=True)

print("\n▶ STEP_1 — Téléchargement et Prétraitement des données (S&P500, Bitcoin)")

for tk in tickers:
    df = yf.download(tk, start=start, end=end, interval=interval, progress=False)

    if df is None or df.empty:
        print(f" Aucune donnée téléchargée pour {tk}", file=sys.stderr)
        continue

    # Préparation des colonnes
    df = df.reset_index().rename(columns=str.lower).sort_values("date")

    # Calcul des rendements logarithmiques (tu l’as déjà mais on le garde pour cohérence)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["rv20"] = df["log_return"].rolling(window=20).std()
    df.dropna(inplace=True)

    # --- Tests de stationnarité (ADF / KPSS) ---
    adf_result = adfuller(df["log_return"])
    adf_p = adf_result[1]

    kpss_result = kpss(df["log_return"], nlags="auto")
    kpss_p = kpss_result[1]

    print(f"\n{tk}")
    print(f"  → ADF p-value  : {adf_p:.4f} ({'stationnaire' if adf_p < 0.05 else 'non stationnaire'})")
    print(f"  → KPSS p-value : {kpss_p:.4f} ({'non stationnaire' if kpss_p < 0.05 else 'stationnaire'})")

    # --- Graphiques ---
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    ax[0].plot(df["date"], df["close"], label="Prix de clôture", color="steelblue")
    ax[0].set_title(f"{tk} - Prix de clôture")
    ax[1].plot(df["date"], df["log_return"], label="Rendement logarithmique", color="orange")
    ax[1].set_title(f"{tk} - Rendements log.")
    plt.tight_layout()
    plt.savefig(f"figures/{tk.replace('^','')}_overview.png")
    plt.close()


    base = tk.replace("^", "")
    df.to_csv(f"data/raw/{base}.csv", index=False)
    df.to_parquet(f"data/processed/{base}.parquet", index=False)

print("\n Étape 1 terminée : données nettoyées, rendements et tests de stationnarité enregistrés.")
print(" Fichiers sauvegardés dans : data/raw, data/processed et figures/")
