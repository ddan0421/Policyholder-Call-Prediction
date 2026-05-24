import numpy as np
from sklearn.metrics import root_mean_squared_error

def gini(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Sort by predicted value, descending (highest-risk first)
    order = np.argsort(y_pred)[::-1]
    y_true = y_true[order]

    n = len(y_true)
    cum_y = np.cumsum(y_true)

    gini_sum = cum_y.sum() / cum_y[-1] - (n + 1) / 2
    return gini_sum / n


def normalized_gini(y_true, y_pred):
    return gini(y_true, y_pred) / gini(y_true, y_true)

def statsmodel_report(label, y_true, y_pred, aic):
    rmse = root_mean_squared_error(y_true, y_pred)
    norm_gini = normalized_gini(y_true, y_pred)
    print(f"{label}: RMSE={rmse:.4f}  AIC={aic:.2f}  NormalizedGini={norm_gini:.4f}")
