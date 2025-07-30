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




###################################################################################################
#                                   Binary Classification Algorithms                              #
###################################################################################################

#----------------------------------------------------------------------------#
#                           Binary Logistic Regression                       #
#----------------------------------------------------------------------------#

"""
- Binary logistic regression with a callback function for monitoring the optimization process
- Make sure X has a intercept/constant variable!
- e.g. if X is divided into X_train and X_test
- X_train = sm.add_constant(X_train)
- X_test = sm.add_constant(X_test)
- e.g. if X is NOT divided into X_train and X_test
- X = sm.add_constant(X)
"""

def sm_logit(X, y, method="ncg", verbose=True):
    X = sm.add_constant(X, has_constant="skip")

    # Define the logit model
    logit = sm.Logit(y, X)

    # Extract feature names
    feature_names = logit.exog_names
    callback_func = callback(feature_names)
    
    # Fit the model with the generalizable callback
    model = logit.fit(method=method,
                      disp=True,  # To display optimization messages
                      maxiter=1000,
                      callback=callback_func)
    
    if verbose:
        # Display final model parameters
        print("\nModel final parameters:")
        print(model.params)
        print("\nModel fitting p-values:")
        print(model.pvalues)
        
        # Save the summary to a text file
        summary_path = f"logit_model_summary.txt"
        with open(summary_path, "w") as file:
            file.write(model.summary().as_text())
        print(f"Model summary saved to {summary_path}\n")
    
    # # Optionally remove unnecessary data to reduce memory usage
    # model.remove_data()
    
    return model

#----------------------------------------------------------------------------#
#                  Constrained Binary Logistic Regression                    #
#----------------------------------------------------------------------------#

def constrained_sm_logit(X, y, logit_result, thresh, verbose=True):
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
        
        # Save the summary to a text file
        summary_path = f"constrained_model_summary.txt"
        with open(summary_path, "w") as file:
            file.write(model.summary().as_text())
        print(f"Constrained Model summary saved to {summary_path}\n")

    return model


#----------------------------------------------------------------------------#
#             Binary Logistic Regression with with Scaling Trick             #
#----------------------------------------------------------------------------#


def sm_logit_scale(X, y, method="ncg", verbose=True):
    """
    Fit a binary logistic regression model with scaling.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series or np.array): Binary target variable.
        method (str): Optimization method for fitting the model.
        verbose (bool): If True, prints progress and model summary.

    Returns:
        modelobj: Fitted binary logistic regression model.
    """
    if verbose:
        print("Fitting Binary Logistic Regression (Logit)...")
        start_time = time.perf_counter()

    X = sm.add_constant(X, has_constant="skip")

    # Step 1: Fit a scaled version -- much faster
    scaler = X.std(axis=0)
    scaler.iloc[0] = 1.0  # Prevent scaling of the intercept if present
    X_s = X.div(scaler, axis='columns')
    
    logit = sm.Logit(y, X_s)

    # Extract feature names
    feature_names = logit.exog_names
    callback_func = callback(feature_names)

    modelobj = logit.fit(method=method,
                         tol=1e-8,
                         maxiter=1000,
                         callback=callback_func)
    start_params = modelobj.params.div(scaler, axis=0).T.to_numpy()

    # Step 2: Fit using the original data
    logit = sm.Logit(y, X)
    modelobj = logit.fit(method=method,
                         tol=1e-8,
                         start_params=start_params,
                         maxiter=1000)

    if verbose:
        duration = time.perf_counter() - start_time
        print(f"Model fitting time: {duration:.2f} seconds")

        # Print model summary
        print(modelobj.summary())
        # Print p-values
        print("\nModel fitting p-values:")
        print(modelobj.pvalues)
    
    return modelobj

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









###################################################################################################
#                                   Statsmodels Classification Evaluation                         #
###################################################################################################

def evaluate_classification_model(model, X_test, y_test, model_name, regression_type="multinomial", library="statsmodels"):
    """
    Evaluates the given model using test data and prints AIC, BIC, and other metrics.
    
    Parameters:
        model: Fitted model object.
        X_test: Test dataset (features).
        y_test: Test dataset (target).
        model_name: Name of the model for display purposes.
        regression_type: "multinomial" or "binary" to specify the type of logistic regression.
        library: "sklearn" or "statsmodels" to specify the library used for the model.
    """
    # Predict probabilities and classes
    if library == "statsmodels":
        predicted_probs = model.predict(X_test)
        if regression_type == "multinomial":
            predicted_classes = np.argmax(predicted_probs, axis=1)
        elif regression_type == "binary":
            predicted_classes = np.where(predicted_probs >= 0.5, 1, 0)
        else:
            raise ValueError("Invalid regression_type. Must be 'multinomial' or 'binary'.")
    elif library == "sklearn":
        predicted_probs = model.predict_proba(X_test)
        if regression_type == "multinomial":
            predicted_classes = np.argmax(predicted_probs, axis=1)
        elif regression_type == "binary":
            predicted_classes = (predicted_probs[:, 1] > 0.5).astype(int)
        else:
            raise ValueError("Invalid regression_type. Must be 'multinomial' or 'binary'.")
    else:
        raise ValueError("Invalid library. Must be 'sklearn' or 'statsmodels'.")
    
    # Evaluate metrics common to both types
    accuracy = accuracy_score(y_test, predicted_classes)
    conf_matrix = confusion_matrix(y_test, predicted_classes)
    class_report = classification_report(y_test, predicted_classes)
    logloss = log_loss(y_test, predicted_probs)

    # Metrics specific to binary classification
    if regression_type == "binary":
        precision = precision_score(y_test, predicted_classes)
        recall = recall_score(y_test, predicted_classes)
        f1 = f1_score(y_test, predicted_classes)
        roc_auc = roc_auc_score(y_test, predicted_probs[:, 1] if library == "sklearn" else predicted_probs)
    else:
        precision = recall = f1 = roc_auc = "Not Applicable"
    
    # AIC and BIC calculations (only for statsmodels)
    if library == "statsmodels":
        try:
            aic = model.aic
            bic = model.bic
        except AttributeError:
            aic = bic = "Not Applicable"
    else:
        aic = bic = "Not Applicable"

    # Print results
    print(f"Evaluation for {model_name} ({regression_type.capitalize()} Classification):")
    print(f"AIC: {aic}")
    print(f"BIC: {bic}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Log-Loss: {logloss:.4f}")
    if regression_type == "binary":
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)
    print("-" * 50)
