# l1_trend_filtering

time-series project l1 trend filtering

TODO :

- mettre dans des scripts à part les fonctions de visualisation, création de données, etc. 
- chaque fichier python ne doit contenir qu’une seule méthode
- donner les temps d’exécution pour plusieurs longueurs de signaux
- faire un readme qui décrit les librairies à installer, comment lancer le code sur un exemple simple, les paramètres des fonctions.

J’ai quelques questions/pistes d’amélioration :
- il y a plusieurs algos implémentés. Lequel correspond à celui de l’article ?
    L'algorithme décrit dans l'article est TODO
- est-ce que ça marche pour des signaux multivariés ?
    Oui, voir TODO
- il faut afficher les ruptures sur les plots que vous montrer, pour voir s’il y a beaucoup ou non.
    TODO avec le package ruptures
- y a-t-il une heuristique pour choisir la pénalité ?
    Pas pour le moment, l'article n'évoque pas d'heuristique pour ce choix.
- est-il possible d’implémenter le trend filtering en utilisant que des librairies simples (Numpy, scipy, scikit-learn, etc.) ?
    Oui, nous avons seulement utilisé numpy, cvxpy et scipy
