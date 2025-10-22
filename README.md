# Volatility Forecasting: GARCH vs LSTM

Projet réalisé dans le cadre du cours **Machine Learning – INSEA S5**  
Encadré par **M. Hicham Janati**


## Objectif du projet

Ce projet vise à comparer deux approches de prévision de la volatilité financière :

- **GARCH (Generalized AutoRegressive Conditional Heteroskedasticity)** : modèle économétrique paramétrique classique, fondé sur la dépendance conditionnelle de la variance.
- **LSTM (Long Short-Term Memory)** : réseau de neurones récurrent capable de modéliser des dépendances temporelles longues et des relations non linéaires.

L’objectif principal est de mesurer la performance prédictive de ces deux approches sur différentes séries financières (indices boursiers et cryptomonnaies) et d’identifier leurs forces et limites respectives.



##  Données utilisées

Les données de prix journaliers sont téléchargées via la librairie `yfinance`.  
Deux séries principales seront analysées :

- **S&P 500 (SPX)** : indice large du marché actions américain, relativement stable.  
- **BTC/USD** : actif à forte volatilité, représentatif des marchés non stationnaires.

Les rendements logarithmiques sont calculés comme :
\[
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
\]
et serviront de base à la modélisation de la volatilité.


## Méthodologie prévue

1. **Prétraitement des données**  
   - Téléchargement, nettoyage et transformation en rendements.  
   - Tests de stationnarité (ADF), normalisation.  

2. **Modélisation GARCH**  
   - Estimation des modèles GARCH(1,1), EGARCH et GJR-GARCH via la librairie `arch`.  
   - Validation par analyse des résidus et tests d’autocorrélation.  

3. **Modélisation LSTM**  
   - Construction d’un réseau LSTM univarié avec fenêtre glissante.  
   - Optimisation des hyperparamètres (nombre de neurones, taux d’apprentissage, epochs).  

4. **Évaluation comparative**  
   - Mesures : RMSE, MAE, MAPE, log-vraisemblance.  
   - Test de **Diebold–Mariano (1995)** pour la comparaison statistique des prévisions.  
   - Visualisation graphique des volatilités prévues.



## Structure du projet (prévue)
```
volatility-forecasting-garch-lstm/
│
├── data/  Données téléchargées (yfinance)
│
├── notebooks/  Analyses et modèles
│ ├── 01_data_exploration.ipynb
│ ├── 02_garch_model.ipynb
│ └── 03_lstm_model.ipynb
│
├── src/  Scripts Python réutilisables
│ ├── garch_model.py
│ ├── lstm_model.py
│ └── evaluation.py
│
├── figures/ Graphiques générés
│
├── report/ Rapports et livrables PDF
│ └── Project_Report_RGD.pdf
│
├── requirements.txt  Librairies nécessaires
└── README.md  Description du projet
```
## Équipe

- **Marwa TAKATRI (DS)**  
- **Salma EN-NASRY (DS)**  
- **Basma REGRAGUI (SE)**
- **Salma EL ALAMI (SE)**



## Technologies utilisées

- **Langage** : Python 3.11  
- **Librairies principales** : `numpy`, `pandas`, `matplotlib`, `arch`, `tensorflow`, `statsmodels`, `yfinance`, `scikit-learn`  
- **Versioning & Collaboration** : GitHub  





##  Document du projet

📄 [**Project Statement – Volatility Forecasting: GARCH vs LSTM**](./report/Project_Report_RGD.pdf)

