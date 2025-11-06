import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model


# ───────────────────────────────────────────────
# 1️⃣ Fonction : Chargement des données
# ───────────────────────────────────────────────
def charger_donnees(chemin_fichier):
    """Charge les données de rendements depuis un fichier CSV."""
    data = pd.read_csv(chemin_fichier)
    if "log_return" not in data.columns:
        raise ValueError("⚠️ La colonne 'log_return' est manquante.")
    data.dropna(subset=["log_return"], inplace=True)
    return data


# ───────────────────────────────────────────────
# 2️⃣ Fonction : Entraînement du modèle GARCH
# ───────────────────────────────────────────────
def entrainer_modele(rendements, type_modele="GARCH"):
    """Entraîne un modèle de volatilité selon le type choisi (GARCH, EGARCH ou GJR-GARCH)."""
    rendements_pct = rendements * 100  # pour plus de stabilité numérique

    if type_modele == "GARCH":
        modele = arch_model(rendements_pct, vol="Garch", p=1, q=1, mean="Zero", dist="normal")

    elif type_modele == "EGARCH":
        modele = arch_model(rendements_pct, vol="EGARCH", p=1, q=1, mean="Zero", dist="normal")

    elif type_modele == "GJR-GARCH":
        modele = arch_model(rendements_pct, vol="GARCH", p=1, q=1, o=1, mean="Zero", dist="normal")

    else:
        raise ValueError("Type de modèle inconnu. Choisir : 'GARCH', 'EGARCH' ou 'GJR-GARCH'.")

    resultat = modele.fit(disp="off")
    return resultat


# ───────────────────────────────────────────────
# 3️⃣ Fonction : Visualisation de la volatilité
# ───────────────────────────────────────────────
def tracer_volatilite(data, resultat, nom_ticker, type_modele, dossier_fig="figures"):
    """Trace et sauvegarde la volatilité conditionnelle estimée."""
    vol = resultat.conditional_volatility
    plt.figure(figsize=(10, 5))
    plt.plot(data["date"], vol, color="purple", linewidth=1.3)
    plt.title(f"{nom_ticker} - Volatilité conditionnelle estimée ({type_modele})")
    plt.xlabel("Date")
    plt.ylabel("Volatilité (%)")
    plt.grid(True)
    os.makedirs(dossier_fig, exist_ok=True)
    chemin_fig = os.path.join(dossier_fig, f"{nom_ticker}_{type_modele}_volatilite.png")
    plt.savefig(chemin_fig, dpi=300)
    plt.close()
    print(f"✅ Figure enregistrée : {chemin_fig}")


# ───────────────────────────────────────────────
# 4️⃣ Fonction : Sauvegarde des résultats
# ───────────────────────────────────────────────
def sauvegarder_resultats(data, resultat, nom_ticker, type_modele, dossier_data="data/processed"):
    """Sauvegarde la volatilité conditionnelle et les paramètres estimés."""
    os.makedirs(dossier_data, exist_ok=True)
    data[f"vol_{type_modele.lower()}"] = resultat.conditional_volatility
    chemin_csv = os.path.join(dossier_data, f"{nom_ticker}_{type_modele}_resultats.csv")
    data.to_csv(chemin_csv, index=False)
    print(f"💾 Résultats sauvegardés : {chemin_csv}")
    print("\n📊 Paramètres estimés :")
    print(resultat.params)
    print("-" * 50)


# ───────────────────────────────────────────────
# 5️⃣ Fonction principale : exécution complète
# ───────────────────────────────────────────────
def main():
    tickers = ["GSPC", "BTC-USD"]
    modeles = ["GARCH", "EGARCH", "GJR-GARCH"]

    for tk in tickers:
        chemin = f"data/raw/{tk}.csv"
        if not os.path.exists(chemin):
            print(f"⚠️ Fichier introuvable : {chemin}")
            continue

        print(f"\n🔹 Ticker en cours : {tk}")
        data = charger_donnees(chemin)

        for m in modeles:
            print(f"\n--- Modèle {m} ---")
            resultat = entrainer_modele(data["log_return"], m)
            tracer_volatilite(data, resultat, tk, m)
            sauvegarder_resultats(data, resultat, tk, m)

    print("\n🎯 Étape 2 terminée : tous les modèles ont été estimés et sauvegardés.")


# ───────────────────────────────────────────────
if __name__ == "__main__":
    main()
