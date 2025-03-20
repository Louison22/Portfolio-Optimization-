###################### Import à ne pas modifier ###################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
##################################################################################

# Chargement des données
npzfile = np.load("dataExamen.npz", allow_pickle=True)
prices = npzfile["mPrice"]
dates = npzfile["vDates"]
names = npzfile["aNames"]

# Conversion des prix en DataFrame pour une manipulation plus facile
df_prices = pd.DataFrame(prices, columns=names, index=pd.to_datetime(dates))

# Calcul des rendements logarithmiques
df_prices_filled = df_prices.ffill()  # Remplissage des valeurs manquantes
returns = np.log(df_prices_filled / df_prices_filled.shift(1)).dropna()

# Fonction pour l'optimisation de portefeuille
def optimize_portfolio(returns, method='min_variance'):
    n_assets = returns.shape[1]
    cov_matrix = returns.cov().values
    expected_returns = returns.mean().values
    ones = np.ones(n_assets)

    if method == 'min_variance':
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        weights = inv_cov_matrix @ ones / (ones.T @ inv_cov_matrix @ ones)
    elif method == 'max_sharpe':
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        weights = inv_cov_matrix @ expected_returns / (ones.T @ inv_cov_matrix @ expected_returns)
    elif method == 'equal_weight':
        weights = np.ones(n_assets) / n_assets
    elif method == 'black_litterman':
        weights = calculate_black_litterman_weights(returns)
    else:
        raise ValueError("Méthode non reconnue")

    return weights

# Fonction pour calculer les poids Black-Litterman
def calculate_black_litterman_weights(returns):
    delta = 2.5  # Paramètre d'aversion au risque
    tau = 0.025  # Paramètre d'incertitude du CAPM prior
    market_cap_weights = np.ones(len(names)) / len(names)  # Poids égaux pour simplifier
    cov_matrix = returns.cov().values
    equilibrium_returns = delta * np.dot(cov_matrix, market_cap_weights)

    # Vues de l'investisseur (exemple simplifié)
    P = np.array([[0, 0, 0, 1, 0, -0.5, -0.5] + [0] * (len(names) - 7)])  # Vue sur 7 actifs
    Q = np.array([0.05]).reshape(-1, 1)  # 5% de rendement supplémentaire
    Omega = np.diagflat([0.01])  # Confiance dans la vue

    # Calcul des poids optimaux
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    Pi = np.dot(inv_cov_matrix, equilibrium_returns)
    tau_cov_matrix = tau * cov_matrix
    omega = np.dot(np.dot(P, tau_cov_matrix), P.T) + Omega
    omega_inv = np.linalg.inv(omega)
    BL_weights = np.dot(inv_cov_matrix, Pi + np.dot(np.dot(P.T, omega_inv), (Q - np.dot(P, Pi))))
    BL_weights /= np.sum(BL_weights)  # Normalisation des poids

    return BL_weights

# Calcul des portefeuilles optimisés
weights_min_var = optimize_portfolio(returns, method='min_variance')
weights_max_sharpe = optimize_portfolio(returns, method='max_sharpe')
weights_equal = optimize_portfolio(returns, method='equal_weight')
weights_bl = optimize_portfolio(returns, method='black_litterman')

# Fonction pour calculer les performances des portefeuilles
def portfolio_performance(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return portfolio_return, portfolio_volatility

# Calcul des performances
ret_min_var, vol_min_var = portfolio_performance(weights_min_var, returns)
ret_max_sharpe, vol_max_sharpe = portfolio_performance(weights_max_sharpe, returns)
ret_equal, vol_equal = portfolio_performance(weights_equal, returns)
ret_bl, vol_bl = portfolio_performance(weights_bl, returns) if weights_bl is not None else (None, None)

# Affichage des performances
print("Performance des portefeuilles:")
print(f"Min variance: rendement = {ret_min_var:.2%}, volatilité = {vol_min_var:.2%}")
print(f"Max Sharpe: rendement = {ret_max_sharpe:.2%}, volatilité = {vol_max_sharpe:.2%}")
print(f"Equal weight: rendement = {ret_equal:.2%}, volatilité = {vol_equal:.2%}")
if ret_bl is not None:
    print(f"Black-Litterman: rendement = {ret_bl:.2%}, volatilité = {vol_bl:.2%}")
else:
    print("Black-Litterman: pas de solution")

# Création de la fiche de présentation
plt.figure(figsize=(11, 8.5))  # Format A4

# Graphique des poids des portefeuilles
plt.subplot(2, 2, 1)
bar_width = 0.2
index = np.arange(len(names))
plt.bar(index, weights_min_var, width=bar_width, color='blue', alpha=0.6, label='Min Variance')
plt.bar(index + bar_width, weights_max_sharpe, width=bar_width, color='green', alpha=0.6, label='Max Sharpe')
plt.bar(index + 2*bar_width, weights_equal, width=bar_width, color='red', alpha=0.6, label='Equal Weight')
if weights_bl is not None:
    plt.bar(index + 3*bar_width, weights_bl, width=bar_width, color='orange', alpha=0.6, label='Black-Litterman')
plt.xticks(index + bar_width, names, rotation=90)
plt.ylabel('Poids')
plt.title('Poids des portefeuilles')
plt.legend()

# Tableau des performances
plt.subplot(2, 2, 2)
performance_data = {
    'Portefeuille': ['Min Variance', 'Max Sharpe', 'Equal Weight', 'Black-Litterman'],
    'Rendement Annuel': [ret_min_var, ret_max_sharpe, ret_equal, ret_bl],
    'Volatilité Annuelle': [vol_min_var, vol_max_sharpe, vol_equal, vol_bl]
}
df_performance = pd.DataFrame(performance_data)
plt.table(cellText=df_performance.values, colLabels=df_performance.columns, loc='center')
plt.axis('off')
plt.title('Performances des Portefeuilles')

# Graphique des rendements cumulés
plt.subplot(2, 1, 2)
cumulative_returns = (1 + returns).cumprod()
cumulative_returns.plot(ax=plt.gca(), linewidth=2, alpha=0.8)
plt.legend(loc='upper left', fontsize='small')
plt.title('Rendements Cumulés des Portefeuilles', fontsize=12)
plt.xlabel('Date', fontsize=10)
plt.ylabel('Rendement Cumulé', fontsize=10)

# Ajustement de la mise en page
plt.tight_layout()
plt.savefig("fiche.pdf", format="pdf")  # Sauvegarde de la fiche en PDF
plt.show()
