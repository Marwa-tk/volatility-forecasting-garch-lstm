import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

#CHARGEMENT ET PRÉPARATION DES DONNÉES
spx_data = pd.read_csv('data/sample/GSPC_sample.csv')
spx_data['date'] = pd.to_datetime(spx_data['date'])
spx_data = spx_data.sort_values('date').reset_index(drop=True)

btc_data = pd.read_csv('data/sample/BTC-USD_sample.csv')
btc_data['date'] = pd.to_datetime(btc_data['date'])
btc_data = btc_data.sort_values('date').reset_index(drop=True)

print("OK données chargées")
print("S&P 500:", len(spx_data), "obs")
print("Bitcoin:", len(btc_data), "obs")

#Récuperer les rendements
spx_returns = spx_data['log_return'].dropna()
btc_returns = btc_data['log_return'].dropna()

print("\nS&P 500 rendements:")
print("Moyenne:", spx_returns.mean())
print("Écart-type:", spx_returns.std())

print("\nBitcoin rendements:")
print("Moyenne:", btc_returns.mean())
print("Écart-type:", btc_returns.std())

#GARCH(1,1) - S&P 500
print("\n" + "="*50)
print("GARCH S&P 500")
print("="*50)

model_spx = arch_model(spx_returns, vol='Garch', p=1, q=1)
result_spx = model_spx.fit(disp='off')
print(result_spx.summary())


#GARCH(1,1) - BITCOIN
print("\n" + "="*50)
print("GARCH BITCOIN")
print("="*50)

model_btc = arch_model(btc_returns, vol='Garch', p=1, q=1)
result_btc = model_btc.fit(disp='off')
print(result_btc.summary())

#EGARCH - S&P 500
print("\n" + "="*50)
print("EGARCH S&P 500")
print("="*50)

model_egarch_spx = arch_model(spx_returns, vol='EGarch', p=1, q=1)
result_egarch_spx = model_egarch_spx.fit(disp='off')
print(result_egarch_spx.summary())

#EGARCH - BITCOIN
print("\n" + "="*50)
print("EGARCH BITCOIN")
print("="*50)

model_egarch_btc = arch_model(btc_returns, vol='EGarch', p=1, q=1)
result_egarch_btc = model_egarch_btc.fit(disp='off')
print(result_egarch_btc.summary())

#VALIDATION - TEST RÉSIDUS
print("\n" + "="*50)
print("VALIDATION RÉSIDUS")
print("="*50)

# S&P 500 GARCH
std_res_spx = result_spx.std_resid
lb_test_spx = acorr_ljungbox(std_res_spx**2, lags=10, return_df=True)
p_val_spx = lb_test_spx['lb_pvalue'].mean()
print("S&P 500 GARCH - p-value Ljung-Box:", round(p_val_spx, 4))

# Bitcoin GARCH
std_res_btc = result_btc.std_resid
lb_test_btc = acorr_ljungbox(std_res_btc**2, lags=10, return_df=True)
p_val_btc = lb_test_btc['lb_pvalue'].mean()
print("Bitcoin GARCH - p-value Ljung-Box:", round(p_val_btc, 4))

# S&P 500 EGARCH
std_res_egarch_spx = result_egarch_spx.std_resid
lb_test_egarch_spx = acorr_ljungbox(std_res_egarch_spx**2, lags=10, return_df=True)
p_val_egarch_spx = lb_test_egarch_spx['lb_pvalue'].mean()
print("S&P 500 EGARCH - p-value Ljung-Box:", round(p_val_egarch_spx, 4))

#SAUVEGARDER LES VOLATILITÉS
vol_spx = result_spx.conditional_volatility
vol_btc = result_btc.conditional_volatility

results_spx = pd.DataFrame({
    'date': spx_data['date'].iloc[1:].values,
    'log_return': spx_returns.values,
    'volatility_garch': vol_spx.values
})

results_btc = pd.DataFrame({
    'date': btc_data['date'].iloc[1:].values,
    'log_return': btc_returns.values,
    'volatility_garch': vol_btc.values
})

results_spx.to_csv('data/sample/GSPC_volatility_garch.csv', index=False)
results_btc.to_csv('data/sample/BTCUSD_volatility_garch.csv', index=False)

print("OK volatilités sauvegardées")

#Visualisation
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# S&P 500 rendements
axes[0, 0].plot(spx_data['date'].iloc[1:], spx_returns, alpha=0.6, color='blue')
axes[0, 0].set_title('S&P 500 - Rendements')
axes[0, 0].grid(alpha=0.3)

# S&P 500 volatilité
axes[0, 1].plot(spx_data['date'].iloc[1:], vol_spx.values, linewidth=1.5, color='darkblue')
axes[0, 1].set_title('S&P 500 - Volatilité GARCH')
axes[0, 1].grid(alpha=0.3)

# Bitcoin rendements
axes[1, 0].plot(btc_data['date'].iloc[1:], btc_returns, alpha=0.6, color='orange')
axes[1, 0].set_title('Bitcoin - Rendements')
axes[1, 0].grid(alpha=0.3)

# Bitcoin volatilité
axes[1, 1].plot(btc_data['date'].iloc[1:], vol_btc.values, linewidth=1.5, color='darkorange')
axes[1, 1].set_title('Bitcoin - Volatilité GARCH')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/garch_volatility.png', dpi=300, bbox_inches='tight')
print("OK figure sauvegardée")
plt.show()