___

# l1 Trend Filtering

MVA time-series course project l1 trend filtering with Charles Truong.

Original paper : https://web.stanford.edu/~boyd/papers/pdf/l1_trend_filter.pdf

Students : 

- Baudouin de Monicault baudouin.de-monicault@polytechnique.edu

- Thomas Li thomas.li9162@gmail.com

___

**Required libraries :** numpy, cvxpy and scipy

**Simple examples :** see demo notebook : 



**Execution time** of filters.univariate.l1_trend_filter with respect to the length of the signal :
![time_execution](https://github.com/bobmnc/l1_trend_filtering/assets/96530384/69fd24d1-1495-4db0-b5c5-947d1a6f311a)




**FAQ :**
- il y a plusieurs algos implémentés. Lequel correspond à celui de l’article ? \
    **L'algorithme principal décrit dans l'article est filters.univariate.l1_trend_filter.py**
- est-ce que ça marche pour des signaux multivariés ? \
    **Oui, nous avons implémenté la version multivarié de l1_trend_filter décrit dans la section 7.5 du papier original dans filters.multivariate.l1_trend_filter_multivariate.py**
- il faut afficher les ruptures sur les plots que vous montrer, pour voir s’il y a beaucoup ou non. \
    **Voir les plots au dessus**
- y a-t-il une heuristique pour choisir la pénalité ? \
    **Nous avons crée une heuristique simple pour choisir la pénalité qui semble plutôt bien marcher sur nos examples dans utils.get_heuristic_lambda.py**
- est-il possible d’implémenter le trend filtering en utilisant que des librairies simples (Numpy, scipy, scikit-learn, etc.) ? \
    **Oui, nous avons seulement utilisé numpy, cvxpy et scipy**
