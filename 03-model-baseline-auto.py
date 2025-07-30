import pandas as pd
import numpy as np
import models
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import root_mean_squared_error
import statsmodels.api as sm
"""
Two simple models to explore 
- Simple OLS regression model
- Ridge and Lasso Regression models 
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor 
- LightGBM Regressor
- CatBoost Regressor 
"""

random_state = 42

############################################## OLS Regression Model ######################################################
X_train = pd.read_csv("data/model_data_auto/X_train.csv")
X_val = pd.read_csv("data/model_data_auto/X_val.csv")
y_train = pd.read_csv("data/model_data_auto/y_train.csv").squeeze()
y_val = pd.read_csv("data/model_data_auto/y_val.csv").squeeze()
X_test = pd.read_csv("data/model_data_auto/X_test.csv")

# Transform numerical features if needed
def transform_features(X):
    log_features = ["ann_prm_amt"]
    cbrt_features = ["tenure_at_snapshot"]
    data = X.copy()
    data[log_features] = np.log1p(data[log_features])
    data[cbrt_features] = np.cbrt(data[cbrt_features])

    return data 

def encode(X):
    data = X.copy()
    cat_cols = ["acq_method_filled", "bi_limit_group", "digital_contact_ind", "geo_group",
                "has_prior_carrier", "household_group", "pay_type_code", "pol_edeliv_ind_filled",
                "prdct_sbtyp_grp", "product_sbtyp", "telematics_ind"]
    
    data_encoded = pd.get_dummies(data, columns=cat_cols, drop_first=True)
    return data_encoded



# Transform numerical features if needed
X_train_transformed = transform_features(X_train)
X_val_transformed = transform_features(X_val)
X_test_transformed = transform_features(X_test)

# Encode categorical features 
X_train_encoded = encode(X_train_transformed)
X_val_encoded = encode(X_val_transformed)
X_test_encoded = encode(X_test_transformed)

X_val_encoded = X_val_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

bool_columns_train = X_train_encoded.select_dtypes(include="bool").columns
bool_columns_val = X_val_encoded.select_dtypes(include="bool").columns
bool_columns_test = X_test_encoded.select_dtypes(include="bool").columns

X_train_encoded[bool_columns_train] = X_train_encoded[bool_columns_train].astype("int8")
X_val_encoded[bool_columns_val] = X_val_encoded[bool_columns_val].astype("int8")
X_test_encoded[bool_columns_test] = X_test_encoded[bool_columns_test].astype("int8")

# Standardize numerical features
numerical_variables = ["12m_call_history", "ann_prm_amt", "home_lot_sq_footage", "household_policy_counts", "newest_veh_age",
                       "tenure_at_snapshot"]
scaler = StandardScaler()
X_train_encoded[numerical_variables] = scaler.fit_transform(X_train_encoded[numerical_variables])
X_val_encoded[numerical_variables] = scaler.transform(X_val_encoded[numerical_variables])
X_test_encoded[numerical_variables] = scaler.transform(X_test_encoded[numerical_variables])


X_train = sm.add_constant(X_train_encoded)
X_val = sm.add_constant(X_val_encoded)
X_test = sm.add_constant(X_test_encoded)

ols_model = models.sm_ols(X_train, y_train)
print(ols_model.summary())

glm_lr_model = models.sm_glm_gaussian(X_train, y_train)
print(glm_lr_model.summary())

con_ols_model = models.constrained_sm_glm_gaussian(X_train, y_train, glm_lr_model, 0.05)
print(con_ols_model.summary())


# Evaluation
y_pred = ols_model.predict(X_val)
rmse = root_mean_squared_error(y_val, y_pred)
print(f"RMSE on validation set for OLS: {rmse}")

y_pred_glm = glm_lr_model.predict(X_val)
rmse_glm = root_mean_squared_error(y_val, y_pred_glm)
print(f"RMSE on validation set for GLM (Gaussian): {rmse_glm}")

y_pred_con = con_ols_model.predict(X_val)
rmse_con = root_mean_squared_error(y_val, y_pred_con)
print(f"RMSE on validation set for Constrained GLM (Gaussian): {rmse_con}")

"""
Summary:
using Gaussian for count data (call_counts) is not ideal
"""


############################################## Regularized Regression Model ######################################################
X_train = pd.read_csv("data/model_data_auto/X_train.csv")
X_val = pd.read_csv("data/model_data_auto/X_val.csv")
y_train = pd.read_csv("data/model_data_auto/y_train.csv").squeeze()
y_val = pd.read_csv("data/model_data_auto/y_val.csv").squeeze()
X_test = pd.read_csv("data/model_data_auto/X_test.csv")

# Transform numerical features if needed
def transform_features(X):
    log_features = ["ann_prm_amt"]
    cbrt_features = ["tenure_at_snapshot"]
    data = X.copy()
    data[log_features] = np.log1p(data[log_features])
    data[cbrt_features] = np.cbrt(data[cbrt_features])

    return data 

# No need to drop the first category in one-hot encoding for regularized regression models
def encode(X):
    data = X.copy()
    cat_cols = ["acq_method_filled", "bi_limit_group", "digital_contact_ind", "geo_group",
                "has_prior_carrier", "household_group", "pay_type_code", "pol_edeliv_ind_filled",
                "prdct_sbtyp_grp", "product_sbtyp", "telematics_ind"]
    
    data_encoded = pd.get_dummies(data, columns=cat_cols, drop_first=False)
    return data_encoded



# Transform numerical features if needed
X_train_transformed = transform_features(X_train)
X_val_transformed = transform_features(X_val)
X_test_transformed = transform_features(X_test)

# Encode categorical features 
X_train_encoded = encode(X_train_transformed)
X_val_encoded = encode(X_val_transformed)
X_test_encoded = encode(X_test_transformed)

X_val_encoded = X_val_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

bool_columns_train = X_train_encoded.select_dtypes(include="bool").columns
bool_columns_val = X_val_encoded.select_dtypes(include="bool").columns
bool_columns_test = X_test_encoded.select_dtypes(include="bool").columns

X_train_encoded[bool_columns_train] = X_train_encoded[bool_columns_train].astype("int8")
X_val_encoded[bool_columns_val] = X_val_encoded[bool_columns_val].astype("int8")
X_test_encoded[bool_columns_test] = X_test_encoded[bool_columns_test].astype("int8")

# Standardize numerical features
numerical_variables = ["12m_call_history", "ann_prm_amt", "home_lot_sq_footage", "household_policy_counts", "newest_veh_age",
                       "tenure_at_snapshot"]
scaler = StandardScaler()
X_train_encoded[numerical_variables] = scaler.fit_transform(X_train_encoded[numerical_variables])
X_val_encoded[numerical_variables] = scaler.transform(X_val_encoded[numerical_variables])
X_test_encoded[numerical_variables] = scaler.transform(X_test_encoded[numerical_variables])


X_train = X_train_encoded.copy()
X_val = X_val_encoded.copy()
X_test = X_test_encoded.copy()

############################# Ridge Regression #############################
cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
ridge = Ridge()

param_grid = {
    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0] 
}

gs_ridge = GridSearchCV(estimator=ridge,
                        param_grid=param_grid,
                        scoring="neg_root_mean_squared_error", 
                        cv=cv,
                        n_jobs=-1,
                        refit=True)

gs_ridge.fit(X_train, y_train)

print("10-Fold CV RMSE (log-transformed scale):", -gs_ridge.best_score_) 
print("Optimal Parameter:", gs_ridge.best_params_)
print("Optimal Estimator:", gs_ridge.best_estimator_)

final_model_ridge = gs_ridge.best_estimator_

############################# Lasso Regression #############################
cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
lasso = Lasso()

param_grid = {
    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
}

gs_lasso = GridSearchCV(estimator=lasso,
                        param_grid=param_grid,
                        scoring="neg_root_mean_squared_error", 
                        cv=cv,
                        n_jobs=-1,
                        refit=True)

gs_lasso.fit(X_train, y_train)

print("10-Fold CV RMSE:", -gs_lasso.best_score_) 
print("Optimal Parameter:", gs_lasso.best_params_)
print("Optimal Estimator:", gs_lasso.best_estimator_)

final_model_lasso = gs_lasso.best_estimator_

# Extract the selected features based on non-zero coefficients from Lasso regression
selected_features_lasso = X_train.columns[final_model_lasso.coef_.flatten() != 0]
print("Selected features for Lasso:")
print(selected_features_lasso)



# Evaluation
y_pred = final_model_ridge.predict(X_val)
rmse = root_mean_squared_error(y_val, y_pred)
print(f"RMSE on validation set for Ridge: {rmse}")

y_pred_glm = final_model_lasso.predict(X_val)
rmse_glm = root_mean_squared_error(y_val, y_pred_glm)
print(f"RMSE on validation set for LASSO: {rmse_glm}")

"""
Summary:
Regularized regression models still use assume Gaussian distribution for the target variable
using Gaussian for count data (call_counts) is not ideal
"""


############################################## Decision Tree Regressor Model ############################################################
X_train = pd.read_csv("data/model_data_auto/X_train.csv")
X_val = pd.read_csv("data/model_data_auto/X_val.csv")
y_train = pd.read_csv("data/model_data_auto/y_train.csv").squeeze()
y_val = pd.read_csv("data/model_data_auto/y_val.csv").squeeze()
X_test = pd.read_csv("data/model_data_auto/X_test.csv")


# No need to drop the first category in one-hot encoding for regularized regression models
def encode(X):
    data = X.copy()
    cat_cols = ["acq_method_filled", "bi_limit_group", "digital_contact_ind", "geo_group",
                "has_prior_carrier", "household_group", "pay_type_code", "pol_edeliv_ind_filled",
                "prdct_sbtyp_grp", "product_sbtyp", "telematics_ind"]
    
    data_encoded = pd.get_dummies(data, columns=cat_cols, drop_first=False)
    return data_encoded


# Encode categorical features 
X_train_encoded = encode(X_train)
X_val_encoded = encode(X_val)
X_test_encoded = encode(X_test)

X_val_encoded = X_val_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

bool_columns_train = X_train_encoded.select_dtypes(include="bool").columns
bool_columns_val = X_val_encoded.select_dtypes(include="bool").columns
bool_columns_test = X_test_encoded.select_dtypes(include="bool").columns

X_train_encoded[bool_columns_train] = X_train_encoded[bool_columns_train].astype("int8")
X_val_encoded[bool_columns_val] = X_val_encoded[bool_columns_val].astype("int8")
X_test_encoded[bool_columns_test] = X_test_encoded[bool_columns_test].astype("int8")



X_train = X_train_encoded.copy()
X_val = X_val_encoded.copy()
X_test = X_test_encoded.copy()


cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
dt = DecisionTreeRegressor(random_state=random_state, criterion="squared_error")

param_grid = {
    "max_depth": [10, 20, 30, 40, None],  
    "min_samples_split": [2, 5, 10, 20],  
    "min_samples_leaf": [1, 2, 5, 10],  
    "min_weight_fraction_leaf": [0.0, 0.01, 0.05],  
}

gs_dt = GridSearchCV(estimator=dt,
                     param_grid=param_grid,
                     scoring="neg_root_mean_squared_error", 
                     cv=cv,
                     n_jobs=-1,
                     refit=True)

gs_dt.fit(X_train, y_train)

print("10-Fold CV RMSE:", -gs_dt.best_score_)  
print("Optimal Parameters:", gs_dt.best_params_)
print("Optimal Estimator:", gs_dt.best_estimator_)

final_model_dt = gs_dt.best_estimator_

selected_features_dt = X_train.columns[np.array(final_model_dt.feature_importances_) > 0]
print("Selected features for Decision Tree:")
print(selected_features_dt)

# Evaluation
y_pred = final_model_dt.predict(X_val)
rmse = root_mean_squared_error(y_val, y_pred)
print(f"RMSE on validation set for Decision Tree: {rmse}")



############################################## Random Forest Tree Regressor Model ############################################################
cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
rf = RandomForestRegressor(random_state=random_state, bootstrap=True)

param_grid = {
    "n_estimators": [50, 100, 200], 
    "max_depth": [3, 5, 10], 
    "min_samples_split": [2, 5],  
    "min_samples_leaf": [1, 2], 
    "max_features": ["sqrt", "log2"],  
}

gs_rf = GridSearchCV(estimator=rf,
                     param_grid=param_grid,
                     scoring="neg_root_mean_squared_error", 
                     cv=cv,
                     n_jobs=-1,
                     refit=True)

gs_rf.fit(X_train, y_train)

print("10-Fold CV RMSE:", -gs_rf.best_score_) 
print("Optimal Parameters:", gs_rf.best_params_)
print("Optimal Estimator:", gs_rf.best_estimator_)

final_model_rf = gs_rf.best_estimator_

selected_features_rf = X_train.columns[np.array(final_model_rf.feature_importances_) > 0]
print("Selected features for Random Forest:")
print(selected_features_rf)

# Evaluation
y_pred = final_model_rf.predict(X_val)
rmse = root_mean_squared_error(y_val, y_pred)
print(f"RMSE on validation set for Random Forest: {rmse}")



############################################## XGBoost Regressor Model ############################################################
cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
xgb_model = xgb.XGBRegressor(random_state=random_state, objective="reg:squarederror")

param_grid = {
    "n_estimators": [180, 200],  
    "learning_rate": [0.07, 0.10], 
    "max_depth": [2, 3, 4],  
    "min_child_weight": [2, 3], 
    "subsample": [0.78, 0.8,],  
    "colsample_bytree": [0.728, 0.75],  
    "reg_alpha": [0, 0.5],  
    "reg_lambda": [0.281, 1]  
}

gs_xgb = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
    refit=True)

gs_xgb.fit(X_train, y_train)

print("10-Fold CV RMSE:", -gs_xgb.best_score_)  
print("Optimal Parameters:", gs_xgb.best_params_)
print("Optimal Estimator:", gs_xgb.best_estimator_)

final_model_xgb = gs_xgb.best_estimator_

selected_features_xgb_final = X_train.columns[np.array(final_model_xgb.feature_importances_) > 0]
print("Selected features for XGBoost:")
print(selected_features_xgb_final)

# Evaluation
y_pred = final_model_xgb.predict(X_val)
rmse = root_mean_squared_error(y_val, y_pred)
print(f"RMSE on validation set for XGBoost: {rmse}")

############################################## LightGBM Regressor Model ############################################################


############################################## CatBoost Regressor Model ############################################################



