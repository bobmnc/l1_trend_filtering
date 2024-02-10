___

# l1 Trend Filtering

MVA time-series course project l1 trend filtering with Charles Truong.

Original paper : https://web.stanford.edu/~boyd/papers/pdf/l1_trend_filter.pdf

___

**Required libraries :** numpy, cvxpy and scipy (and ruptures to display the results) 

**Simple examples :** 

```
# general imports
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ruptures as rpt
sns.set_theme()

from datasets.univariate.create_signals import create_signals
from filters.univariate.l1_trend_filter import l1_trend_filter
from utils.get_heuristic_lambda import get_heuristic_lambda
from utils.get_breakpoints import get_breakpoints

N_signals = 1

# Create synthetic piecewise linear signals with noise
signals, breakpoints_list = create_signals(
    N=N_signals, signal_length=1000, max_slope=0.5, p_trend_change=0.01, noise_level=20.
)
# signals is of shape (N_signals, signal_length)
signal = signals[0]
bkps = breakpoints_list[0]

# Heuristic to choose lambda value
penalty = get_heuristic_lambda(signal) # a positive float
print("heuristic lambda :", penalty)

# get the trend from l1_trend_filtering
trend_l1 = l1_trend_filter(signal, penalty = penalty) # shape (signal_length,)

# get the predicted breakpoints of the filtered trend
pred_bkps = get_breakpoints(trend_l1) # list of breakpoints
print("Number of breakpoints : ", len(pred_bkps))

# display the predicted changepoints and the trend
rpt.display(signal, bkps, pred_bkps)
x_axis = np.arange(len(signal))
plt.plot(x_axis, trend_l1, label = "trend_l1")
plt.legend()
plt.show()
```
heuristic lambda : 37279.079811632226

Number of breakpoints :  4

![output1](https://github.com/bobmnc/l1_trend_filtering/assets/96530384/d67a45cd-7221-4146-ab61-9e0240fbf332)


**See more examples in the demo notebook : demo.ipynb**


**Execution time** of filters.univariate.l1_trend_filter with respect to the length of the signal :

![time_execution](https://github.com/bobmnc/l1_trend_filtering/assets/96530384/69fd24d1-1495-4db0-b5c5-947d1a6f311a)




**FAQ :**
- il y a plusieurs algos implémentés. Lequel correspond à celui de l’article ? \
    **L'algorithme principal décrit dans l'article est filters.univariate.l1_trend_filter.py**
- est-ce que ça marche pour des signaux multivariés ? \
    **Oui, nous avons implémenté la version multivarié de l1_trend_filter décrit dans la section 7.5 du papier original dans filters.multivariate.l1_trend_filter_multivariate.py**
- y a-t-il une heuristique pour choisir la pénalité ? \
    **Nous avons crée une heuristique simple pour choisir la pénalité qui semble plutôt bien marcher sur nos examples dans utils.get_heuristic_lambda.py**
- est-il possible d’implémenter le trend filtering en utilisant que des librairies simples (Numpy, scipy, scikit-learn, etc.) ? \
    **Oui, nous avons seulement utilisé numpy, cvxpy et scipy**
- Est-il possible de supprimer les ruptures associées à des changements de dérivé trop faible (seuillage) ? \
    **Oui nous avons fait ca dans la fonction utils.get_breakpoints.py**
- il faut afficher les ruptures sur les plots que vous montrer, pour voir s’il y a beaucoup ou non. \
    **Examples de plots univariés** :
![example1](https://github.com/bobmnc/l1_trend_filtering/assets/96530384/064c2a4a-fa71-4aaf-819f-51790378579a)
![example2](https://github.com/bobmnc/l1_trend_filtering/assets/96530384/1761e111-c944-4f45-a85c-59f1ba89f2f9)
![example3](https://github.com/bobmnc/l1_trend_filtering/assets/96530384/47419195-de9b-42e4-b359-89fb139e5705)
![example4](https://github.com/bobmnc/l1_trend_filtering/assets/96530384/b5abf007-bd47-4954-a149-929561b9ef94)
![example5](https://github.com/bobmnc/l1_trend_filtering/assets/96530384/107ac487-795d-41e3-a8e9-1b68e2580eee)

**Examples de plot multivarié** pour un signal en dimension 5 :
![multivariate](https://github.com/bobmnc/l1_trend_filtering/assets/96530384/83f9f86c-d1ba-4fcd-b547-89449425f423)



  
