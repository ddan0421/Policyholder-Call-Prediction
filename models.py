import statsmodels.api as sm
import pandas as pd
import numpy as np
from scipy.optimize import LinearConstraint, minimize
from statsmodels.stats.outliers_influence import variance_inflation_factor
from warnings import filterwarnings

filterwarnings("ignore", "divide by zero", category=RuntimeWarning)
filterwarnings("ignore", "invalid value", category=RuntimeWarning)
filterwarnings("ignore", "overflow encountered", category=RuntimeWarning)


###################################################################################################
#                                   Feature Selection Algorithm                                   #
###################################################################################################


#----------------------------------------------------------------------------#
#                            VIF Feature Selection                           #
#----------------------------------------------------------------------------#

def select_features_by_vif(X, threshold=10, verbose=True):
    """
    Remove multicollinear features using Variance Inflation Factor (VIF).
    
    Parameters:
        X : pandas.DataFrame
            DataFrame containing only the independent variables.
        threshold : float, optional
            VIF threshold above which features are iteratively removed. Default is 10.
        verbose : bool, optional
            If True, prints the feature removal process. Default is True.
    
    Returns:
        pd.DataFrame: DataFrame containing the final features and their VIF values.
        list: List of selected features after VIF-based feature selection.
    """
    X = X.copy()
    dropped_features = []

    def calculate_vif(data):
        """Calculate VIF for all features in a DataFrame."""
        data = sm.add_constant(data, has_constant="skip")
        vif_data = pd.DataFrame({
            "Feature": data.columns,
            "VIF": [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
        })
        return vif_data[vif_data["Feature"] != "const"].reset_index(drop=True)
    
    while True:
        vif_data = calculate_vif(X)
        max_vif = vif_data["VIF"].max()
        
        if max_vif < threshold:
            break  # Stop when all VIF values are below the threshold
        
        # Identify and drop the feature with the highest VIF
        feature_to_remove = vif_data.loc[vif_data["VIF"] == max_vif, "Feature"].values[0]
        if verbose:
            print(f"Removed: {feature_to_remove}, VIF: {max_vif}")
        dropped_features.append((feature_to_remove, max_vif))
        X = X.drop(columns=[feature_to_remove])
    
    selected_features = X.columns.tolist()
    if verbose:
        print("Final selected features:", selected_features)
    
    return calculate_vif(X), selected_features



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
#                                   Regression Algorithms                                         #
###################################################################################################

#----------------------------------------------------------------------------#
#                             Linear Regression (OLS)                        #
#----------------------------------------------------------------------------#

def sm_ols(X, y, method="qr", verbose=True):
    X = sm.add_constant(X, has_constant="skip")
    ols = sm.OLS(y,X)
    model = ols.fit(method=method,
                    disp=True,
                    maxiter=1000)
    if verbose:
        # Display final model parameters
        print("\nModel final parameters:")
        print(model.params)
        print("\nModel fitting summary:")
        print(model.summary())

        # Save the summary to a text file
        summary_path = f"ols_model_summary.txt"
        with open(summary_path, "w") as file:
            file.write(model.summary().as_text())
        print(f"Model summary saved to {summary_path}\n")

        # Optionally remove unnecessary data to reduce memory usage
        # model.remove_data()

    return model



def constrained_sm_ols(X, y, ols_result, thresh, method="trust-constr"):
    X = sm.add_constant(X, has_constant="skip")
    n_features = X.shape[1]
    sig = ols_result.pvalues
    sig.reset_index(drop=True, inplace=True)

    constraints_index = np.where(sig > thresh)[0]

    ncons = len(constraints_index)
    A = np.zeros((ncons, n_features))
    A[np.arange(ncons), constraints_index] = 1

    lb = ub = np.zeros(ncons)
    constraints = LinearConstraint(A, lb, ub)


    if method == "SLSQP":
        start_params = ols_result.params.to_numpy(copy=True).ravel(order="F")
    else:
        start_params = ols_result.params.to_numpy(copy=True)
    start_params[constraints_index] = 0

    callback_func = callback(X.columns)

    def objective(params):
        residuals = y - np.dot(X, params)
        return np.sum(residuals**2)
    
    result = minimize(objective, start_params, method=method, constraints=constraints, 
                      callback=callback_func, tol=5e-9, options={"maxiter": 1000, "disp":True})
    
    param_names = X.columns
    optimized_params = pd.Series(result.x, index=param_names)
    
    return optimized_params


#----------------------------------------------------------------------------#
#                    Linear Regression (GLM Gaussian)                        #
#----------------------------------------------------------------------------#

def sm_glm_gaussian(X, y, method="IRLS", verbose=True):
    X = sm.add_constant(X, has_constant="skip")
    glm = sm.GLM(y, X, family=sm.families.Gaussian())
    model = glm.fit(
        method=method,
        maxiter=1000,
        tol=1e-9
    )

    if verbose:
        # Display final model parameters
        print("\nModel final parameters:")
        print(model.params)
        print("\nModel fitting summary:")
        print(model.summary())

        # Save the summary to a text file
        summary_path = f"GLM_Gaussian_summary.txt"
        with open(summary_path, "w") as file:
            file.write(model.summary().as_text())
        print(f"Model summary saved to {summary_path}\n")

    # Optionally remove unnecessary data to reduce memory usage
    # model.remove_data()

    return model

def constrained_sm_glm_gaussian(X, y, glm_gau_result, thresh, verbose=True):
    X = sm.add_constant(X, has_constant="skip")
    n_features = X.shape[1]
    sig = glm_gau_result.pvalues
    sig.reset_index(drop=True, inplace=True)


    constraints_index = np.where(sig > thresh)[0]
    ncons = len(constraints_index)

    R = np.zeros((ncons, n_features))
    R[np.arange(ncons), constraints_index] = 1
    q = np.zeros(ncons)

    start_params = glm_gau_result.params.to_numpy(copy=True)
    start_params[constraints_index] = 0

    constrained_model_glm = sm.GLM(y, X, family=sm.families.Gaussian())
    model = constrained_model_glm.fit_constrained(start_params=start_params,
                                                  constraints=(R,q))
    if verbose:
        # Display final model parameters
        print("\nModel final parameters:")
        print(model.params)
        print("\nModel fitting p-values:")
        print(model.pvalues)
        print("\nModel fitting summary:")
        print(model.summary())

        summary_path = f"GLM_Gaussian_summary_constrained.txt"
        with open(summary_path, "w") as file:
            file.write(model.summary().as_text())
        print(f"Model summary saved to {summary_path}\n")

    # Optionally remove unnecessary data to reduce memory usage
    # model.remove_data()

    return model

#----------------------------------------------------------------------------#
#          Stepwise Feature Selection for OLS Regression                    #
#----------------------------------------------------------------------------#
def ols_stepwise_selection(x_data,
                           y_data, 
                           initial_list  = [], 
                           threshold_in  = 0.01, 
                           threshold_out = 0.05, 
                           verbose       = True):
    """
    Perform a forward-backward feature selection based on p-values from statsmodels.api.OLS.

    Arguments:
        x_data : pandas.DataFrame
            DataFrame with candidate features.
        y_data : array-like
            The target variable.
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

            model = sm.OLS(y_data, sm.add_constant(x_data[included + [new_column]], has_constant="skip")).fit()

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
            model = sm.OLS(y_data, sm.add_constant(x_data[included], has_constant="skip")).fit()

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
