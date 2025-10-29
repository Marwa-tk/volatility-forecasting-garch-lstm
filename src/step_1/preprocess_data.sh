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
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from statsmodels.tsa.stattools import adfuller, kpss

sns.set_style("whitegrid")

tickers = os.getenv("TICKERS","^GSPC,BTC-USD").split(",")
start   = os.getenv("START_DATE","2018-01-01")
end     = os.getenv("END_DATE","2025-01-01")
interval= os.getenv("INTERVAL","1d")

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("figures", exist_ok=True)

for tk in tickers:
    try:
        df = yf.download(tk, start=start, end=end, interval=interval, progress=False)
    except Exception as e:
        print(f"[ERREUR] Impossible de télécharger {tk}: {e}", file=sys.stderr)
        continue

    if df.empty:
        print(f"[AVERTISSEMENT] Aucune donnée pour {tk}", file=sys.stderr)
        continue

    df = df.reset_index()
    df = df.rename(columns=str.lower)
    df = df.sort_values("date")
    
    # Calcul des rendements logarithmiques
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["rv20"] = df["log_return"].rolling(20).std()
    df.dropna(inplace=True)

    # Tests de stationnarité
    adf_p = adfuller(df["log_return"])[1]
    try:
        kpss_p = kpss(df["log_return"], nlags="auto")[1]
    except:
        kpss_p = np.nan

    print(f"\n {tk}")
    print(f"  → ADF p-value  : {adf_p:.4f} ({'stationnaire' if adf_p<0.05 else 'non stationnaire'})")
    print(f"  → KPSS p-value : {kpss_p:.4f} ({'non stationnaire' if kpss_p<0.05 else 'stationnaire'})")

    # Graphiques
    fig, ax = plt.subplots(2,1, figsize=(10,6))
    ax[0].plot(df["date"], df["close"], label="Prix de clôture", color="steelblue")
    ax[0].set_title(f"{tk} - Prix")
    ax[1].plot(df["date"], df["log_return"], label="Rendement log.", color="orange")
    ax[1].set_title(f"{tk} - Rendements logarithmiques")
    plt.tight_layout()
    plt.savefig(f"figures/{tk.replace('^','')}_overview.png")
    plt.close()

    # Sauvegardes
    base = tk.replace("^","")
    df.to_csv(f"data/raw/{base}.csv", index=False)
    df.to_parquet(f"data/processed/{base}.parquet", index=False)

print("\n Étape 1 terminée : données nettoyées, rendements et tests OK.")
PY

green "STEP_1 terminé — fichiers enregistrés dans data/raw, data/processed et figures/"
feat(step_1): preprocessing pipeline (download, log-returns, ADF, KPSS)
