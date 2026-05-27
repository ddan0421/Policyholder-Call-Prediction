import os
import statsmodels.api as sm
import pandas as pd
import numpy as np
from scipy.optimize import LinearConstraint, minimize
from statsmodels.stats.outliers_influence import variance_inflation_factor
from warnings import filterwarnings
import time
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss, precision_score, recall_score, f1_score, roc_auc_score

filterwarnings("ignore", "divide by zero", category=RuntimeWarning)
filterwarnings("ignore", "invalid value", category=RuntimeWarning)
filterwarnings("ignore", "overflow encountered", category=RuntimeWarning)


def save_model_summary(model, model_output_dir, filename):
    os.makedirs(model_output_dir, exist_ok=True)
    summary_path = os.path.join(model_output_dir, filename)
    with open(summary_path, "w") as file:
        file.write(model.summary().as_text())
    print(f"Model summary saved to {summary_path}\n")


###################################################################################################
#                                  Count Regression Algorithms                                    #
###################################################################################################
"""
References:
- Poisson GLM: https://www.statsmodels.org/dev/generated/statsmodels.genmod.generalized_linear_model.GLM.html
- ZIP:         https://www.statsmodels.org/dev/generated/statsmodels.discrete.count_model.ZeroInflatedPoisson.html
- ZINB:        https://www.statsmodels.org/dev/generated/statsmodels.discrete.count_model.ZeroInflatedNegativeBinomialP.html
"""

#----------------------------------------------------------------------------#
#                                Poisson GLM                                 #
#----------------------------------------------------------------------------#

"""
- Poisson GLM with log link for count regression
- Make sure X has an intercept/constant variable!
- e.g. if X is divided into X_train and X_test
- X_train = sm.add_constant(X_train)
- X_test = sm.add_constant(X_test)
- e.g. if X is NOT divided into X_train and X_test
- X = sm.add_constant(X)
"""

def sm_poisson_glm(X, y, model_output_dir, verbose=True):
    X = sm.add_constant(X, has_constant="skip")

    pois_model = sm.GLM(y, X, family=sm.families.Poisson())
    model = pois_model.fit()

    if verbose:
        print(model.summary())
        save_model_summary(model, model_output_dir, "poisson_glm_model_summary.txt")

    return model


#----------------------------------------------------------------------------#
#               Warm-start helper for Zero-Inflated Models                   #
#----------------------------------------------------------------------------#

def zi_warm_start(X, y):
    """
    Compute warm-start parameters for ZIP / ZINB:
    - Inflation logit: logistic regression of (y == 0) on X
    - Count part:      Poisson GLM on positive-count subset (y > 0)

    X must already include the intercept column.
    Returns (infl_start, count_start) as 1-D numpy arrays.
    """
    y = np.asarray(y)
    y_zero = (y == 0).astype(int)

    logit_fit = sm.Logit(y_zero, X).fit(method="bfgs", disp=False)
    infl_start = np.asarray(logit_fit.params)

    mask_pos = y > 0
    pois_pos = sm.GLM(y[mask_pos], X[mask_pos], family=sm.families.Poisson()).fit()
    count_start = np.asarray(pois_pos.params)

    return infl_start, count_start


#----------------------------------------------------------------------------#
#                       Zero-Inflated Poisson (ZIP)                          #
#----------------------------------------------------------------------------#

"""
- Zero-inflated Poisson with feature-driven inflation logit and Poisson count
- Both inflation and count parts use the same X
- start_params: None (default) lets statsmodels choose; pass an array to use
  warm-start parameters (e.g. from zi_warm_start(X, y))
- Make sure X has an intercept/constant variable!
"""

def sm_zip(X, y, model_output_dir, method="bfgs", maxiter=2000, start_params=None, verbose=True):
    X = sm.add_constant(X, has_constant="skip")

    zip_model = sm.ZeroInflatedPoisson(
        endog=y,
        exog=X,
        exog_infl=X,
        inflation="logit",
    )
    model = zip_model.fit(
        start_params=start_params,
        method=method,
        maxiter=maxiter,
        disp=False,
    )

    if verbose:
        print(model.summary())
        save_model_summary(model, model_output_dir, "zip_model_summary.txt")

    return model


#----------------------------------------------------------------------------#
#                Zero-Inflated Negative Binomial (ZINB)                      #
#----------------------------------------------------------------------------#

"""
- Zero-inflated negative binomial with feature-driven inflation logit and NB count
- p=2 corresponds to NB2 parameterization (variance = mu + alpha * mu^2)
- start_params: None (default) lets statsmodels choose; pass an array to use
  warm-start parameters (e.g. from zi_warm_start(X, y))
- Make sure X has an intercept/constant variable!
"""

def sm_zinb(X, y, model_output_dir, method="bfgs", maxiter=2000, p=2, start_params=None, verbose=True):
    X = sm.add_constant(X, has_constant="skip")

    zinb_model = sm.ZeroInflatedNegativeBinomialP(
        endog=y,
        exog=X,
        exog_infl=X,
        inflation="logit",
        p=p,
    )
    model = zinb_model.fit(
        start_params=start_params,
        method=method,
        maxiter=maxiter,
        disp=False,
    )

    if verbose:
        print(model.summary())
        save_model_summary(model, model_output_dir, "zinb_model_summary.txt")

    return model





###################################################################################################
#                      Linear and Logistic Regression Callback Function                           #
###################################################################################################
def callback(feature_names):
    iter = 0
    
    def display_param(xk, res=None):
        nonlocal iter # iter can be used in outer function but not globally
        iter += 1
        print(f"Iteration: {iter}")
        print("Current parameter values:")
        for name, value in zip(feature_names, xk):
            print(f"{name:<30} {value:.6f}")
        print("\n") 

    return display_param


###################################################################################################
#                                   Binary Classification Algorithms                              #
###################################################################################################

#----------------------------------------------------------------------------#
#                           Binary Logistic Regression                       #
#----------------------------------------------------------------------------#

"""
- Binary logistic regression with optional warm-start scaling trick.
- When is_scaled=True (default) X is assumed to already be on a well-conditioned
  scale (e.g. StandardScaler applied in data prep) and the model is fit directly.
- When is_scaled=False X is on its original scale; the function:
    1. Builds a scaler that rescales only continuous numeric columns (binary 0/1
       columns, the intercept, and constant columns are kept as-is so their
       coefficients are not distorted).
    2. Fast-fits on the rescaled X for quick convergence.
    3. Refits on the original X using the rescaled params as warm-start so the
       final coefficients are on the original feature scale.
- Make sure X has an intercept/constant variable!
- e.g. if X is divided into X_train and X_test
- X_train = sm.add_constant(X_train)
- X_test = sm.add_constant(X_test)
- e.g. if X is NOT divided into X_train and X_test
- X = sm.add_constant(X)
"""

def sm_logit(X, y, model_output_dir, method="ncg", is_scaled=True, verbose=True):
    if verbose:
        print("Fitting Binary Logistic Regression (Logit)...")
        start_time = time.perf_counter()

    X = sm.add_constant(X, has_constant="skip")

    if is_scaled:
        # X is already on a well-conditioned scale -- fit directly.
        logit = sm.Logit(y, X)
        feature_names = logit.exog_names
        callback_func = callback(feature_names)
        model = logit.fit(method=method,
                          disp=True,
                          tol=1e-8,
                          maxiter=1000,
                          callback=callback_func)
    else:
        # X is on its original scale -- apply the warm-start scaling trick.
        # Skip binary (0/1) columns, the intercept (all 1s), and constant cols
        # so their coefficients are not distorted.
        scaler = X.std(axis=0).astype(float)
        is_binary = ((X == 0) | (X == 1)).all(axis=0)
        scaler[is_binary] = 1.0
        scaler[scaler == 0] = 1.0
        X_s = X.div(scaler, axis="columns")

        # Step 1: fast fit on rescaled X
        logit_s = sm.Logit(y, X_s)
        feature_names = logit_s.exog_names
        callback_func = callback(feature_names)
        fast = logit_s.fit(method=method,
                           disp=True,
                           tol=1e-8,
                           maxiter=1000,
                           callback=callback_func)

        # Step 2: refit on original X using rescaled params as warm-start
        start_params = (fast.params / scaler).to_numpy()
        logit = sm.Logit(y, X)
        model = logit.fit(method=method,
                          disp=True,
                          tol=1e-8,
                          start_params=start_params,
                          maxiter=1000)

    if verbose:
        duration = time.perf_counter() - start_time
        print(f"Model fitting time: {duration:.2f} seconds\n")
        print(model.summary())
        print("\nModel fitting p-values:")
        print(model.pvalues)
        save_model_summary(model, model_output_dir, "logit_model_summary.txt")

    return model

#----------------------------------------------------------------------------#
#                  Constrained Binary Logistic Regression                    #
#----------------------------------------------------------------------------#

def constrained_sm_logit(X, y, logit_result, thresh, model_output_dir, verbose=True):
    X = sm.add_constant(X, has_constant="skip")

    # setup constraints
    n_features = X.shape[1]
    sig = logit_result.pvalues
    sig.reset_index(drop=True, inplace=True)

    # Vectorized filtering for constraints
    constraints_index = np.where(sig > thresh)[0]

    ncons = len(constraints_index)
    A = np.zeros((ncons, n_features))
    A[np.arange(ncons), constraints_index] = 1  # Vectorized row assignment

    lb = ub = np.zeros(ncons)
    constraints = LinearConstraint(A, lb, ub)

    start_params = logit_result.params.to_numpy(copy=True).ravel(order="F")
    start_params[constraints_index] = 0  # Vectorized zeroing

    callback_fun = callback(X.columns)
    constrained_model_logit = sm.Logit(y, X)
    model = constrained_model_logit.fit(method="minimize",
                            min_method="SLSQP",
                            start_params=start_params,
                            constraints=constraints,
                            ftol=5e-9,
                            callback=callback_fun,
                            maxiter=1000)

    if verbose:
        # Display final model parameters
        print("\nModel final parameters:")
        print(model.params)
        print("\nModel fitting p-values:")
        print(model.pvalues)
        save_model_summary(model, model_output_dir, "constrained_logit_model_summary.txt")

    return model


#----------------------------------------------------------------------------#
#         Stepwise Feature Selection for Logistic Regression                 #
#----------------------------------------------------------------------------#
def lr_stepwise_selection(x_data,
                           y_data, 
                           initial_list  = [], 
                           threshold_in  = 0.01, 
                           threshold_out = 0.05,
                           method        = "ncg", 
                           verbose       = True):
    """
    Perform a forward-backward feature selection based on p-values from statsmodels.api.OLS.

    Arguments:
        x_data : pandas.DataFrame
            DataFrame with candidate features.
        y_data : array-like
            The target binary variable (0 or 1)
        initial_list : list
            List of features (column names of x) to start with.
        threshold_in : float
            Include a feature if its p-value < threshold_in.
        threshold_out : float
            Exclude a feature if its p-value > threshold_out.
        verbose : bool
            Whether to print the sequence of inclusions and exclusions.

    Returns:
        list : The list of selected features.

    Note: Always set threshold_in < threshold_out to avoid infinite loops.
    """
    # setting placeholer list
    included = list(initial_list)


    # looping over each x-feature until there are no more significant p-values
    while True:
        changed = False

        # forward step: adding an x-feature
        excluded = [col for col in x_data.columns if col not in included]
        new_pvals = pd.Series(dtype=float, index=excluded)


        # fitting model with additional candidate feature
        for new_column in excluded:

            model = sm.Logit(y_data, sm.add_constant(x_data[included + [new_column]], has_constant="skip")).fit(method=method, maxiter=1000, disp=False)

            new_pvals[new_column] = model.pvalues[new_column]


        if not new_pvals.empty:
            best_pval = new_pvals.min()
            if best_pval < threshold_in:
                best_feature = new_pvals.idxmin()  # Use idxmin() instead of argmin()
                included.append(best_feature)
                changed = True
                if verbose:
                    print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))


        # backward step: potentially removing an x-feature
        if included:
            model = sm.Logit(y_data, sm.add_constant(x_data[included], has_constant="skip")).fit(method=method, maxiter=1000, disp=False)

            # excluding intercept p-value (first element)
            pvals = model.pvalues.iloc[1:]

            # ensuring the model is not empty
            if not pvals.empty:
                worst_pval = pvals.max()
                if worst_pval > threshold_out:
                    worst_feature = pvals.idxmax()  # Use idxmax() instead of argmax()
                    included.remove(worst_feature)
                    changed = True
                    if verbose:
                        print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))


        # stopping the loop if optimized
        if not changed:
            break


    # returning stepwise model's x-features
    return included

