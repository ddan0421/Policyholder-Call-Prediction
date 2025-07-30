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
import catboost as cb
"""
Two simple models to explore for NonAuto dataset:
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
X_train = pd.read_csv("data/model_data_Nonauto/X_train.csv")
X_val = pd.read_csv("data/model_data_Nonauto/X_val.csv")
y_train = pd.read_csv("data/model_data_Nonauto/y_train.csv").squeeze()
y_val = pd.read_csv("data/model_data_Nonauto/y_val.csv").squeeze()
X_test = pd.read_csv("data/model_data_Nonauto/X_test.csv")

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
    cat_cols = ["acq_method_filled", "channel", "digital_contact_ind", "geo_group",
                "has_prior_carrier", "household_group", "pay_type_code", "pol_edeliv_ind_filled",
                "prdct_sbtyp_grp", "product_sbtyp"]
    
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
numerical_variables = ["12m_call_history", "ann_prm_amt", "home_lot_sq_footage", "household_policy_counts", 
                       "tenure_at_snapshot", "trm_len_mo"]
scaler = StandardScaler()
X_train_encoded[numerical_variables] = scaler.fit_transform(X_train_encoded[numerical_variables])
X_val_encoded[numerical_variables] = scaler.transform(X_val_encoded[numerical_variables])
X_test_encoded[numerical_variables] = scaler.transform(X_test_encoded[numerical_variables])


X_train = sm.add_constant(X_train_encoded)
X_val_ols = sm.add_constant(X_val_encoded)
X_test = sm.add_constant(X_test_encoded)

ols_model = models.sm_ols(X_train, y_train)
print(ols_model.summary())

glm_lr_model = models.sm_glm_gaussian(X_train, y_train)
print(glm_lr_model.summary())

con_ols_model = models.constrained_sm_glm_gaussian(X_train, y_train, glm_lr_model, 0.05)
print(con_ols_model.summary())


############################################## Regularized Regression Model ######################################################
X_train = pd.read_csv("data/model_data_Nonauto/X_train.csv")
X_val = pd.read_csv("data/model_data_Nonauto/X_val.csv")
y_train = pd.read_csv("data/model_data_Nonauto/y_train.csv").squeeze()
y_val = pd.read_csv("data/model_data_Nonauto/y_val.csv").squeeze()
X_test = pd.read_csv("data/model_data_Nonauto/X_test.csv")

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
    cat_cols = ["acq_method_filled", "channel", "digital_contact_ind", "geo_group",
                "has_prior_carrier", "household_group", "pay_type_code", "pol_edeliv_ind_filled",
                "prdct_sbtyp_grp", "product_sbtyp"]
    
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
numerical_variables = ["12m_call_history", "ann_prm_amt", "home_lot_sq_footage", "household_policy_counts", 
                       "tenure_at_snapshot", "trm_len_mo"]
scaler = StandardScaler()
X_train_encoded[numerical_variables] = scaler.fit_transform(X_train_encoded[numerical_variables])
X_val_encoded[numerical_variables] = scaler.transform(X_val_encoded[numerical_variables])
X_test_encoded[numerical_variables] = scaler.transform(X_test_encoded[numerical_variables])


X_train = X_train_encoded.copy()
X_val_reg = X_val_encoded.copy()
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



############################################## Decision Tree Regressor Model ############################################################
X_train = pd.read_csv("data/model_data_Nonauto/X_train.csv")
X_val = pd.read_csv("data/model_data_Nonauto/X_val.csv")
y_train = pd.read_csv("data/model_data_Nonauto/y_train.csv").squeeze()
y_val = pd.read_csv("data/model_data_Nonauto/y_val.csv").squeeze()
X_test = pd.read_csv("data/model_data_Nonauto/X_test.csv")


# No need to drop the first category in one-hot encoding for regularized regression models
def encode(X):
    data = X.copy()
    cat_cols = ["acq_method_filled", "channel", "digital_contact_ind", "geo_group",
                "has_prior_carrier", "household_group", "pay_type_code", "pol_edeliv_ind_filled",
                "prdct_sbtyp_grp", "product_sbtyp"]
    
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
X_val_tree = X_val_encoded.copy()
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

############################################## LightGBM Regressor Model ############################################################
X_train = pd.read_csv("data/model_data_Nonauto/X_train.csv")
X_val_cat = pd.read_csv("data/model_data_Nonauto/X_val.csv")
y_train = pd.read_csv("data/model_data_Nonauto/y_train.csv").squeeze()
y_val = pd.read_csv("data/model_data_Nonauto/y_val.csv").squeeze()
X_test = pd.read_csv("data/model_data_Nonauto/X_test.csv")

cat_cols = ["acq_method_filled", "channel", "digital_contact_ind", "geo_group",
            "has_prior_carrier", "household_group", "pay_type_code", "pol_edeliv_ind_filled",
            "prdct_sbtyp_grp", "product_sbtyp"]


X_train[cat_cols] = X_train[cat_cols].astype("category")
X_val_cat[cat_cols] = X_val_cat[cat_cols].astype("category")


cv = KFold(n_splits=10, shuffle=True, random_state=random_state)

lgbm = lgb.LGBMRegressor(random_state=random_state, objective="regression", verbose=-1)

param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.08, 0.11],          
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "min_child_samples": [10, 20]             
}

gs_lgbm = GridSearchCV(
    estimator=lgbm,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
    refit=True)

gs_lgbm.fit(X_train, y_train, categorical_feature=cat_cols)

print("10-Fold CV RMSE:", -gs_lgbm.best_score_) 
print("Optimal Parameters:", gs_lgbm.best_params_)
print("Optimal Estimator:", gs_lgbm.best_estimator_)

final_model_lgbm = gs_lgbm.best_estimator_

selected_features_lgbm = X_train.columns[np.array(final_model_lgbm.feature_importances_) > 0]
print("Selected features for LightGBM:")
print(selected_features_lgbm)


############################################## CatBoost Regressor Model ############################################################
train_pool = cb.Pool(data=X_train, label=y_train, cat_features=cat_cols)
val_pool = cb.Pool(data=X_val_cat, label=y_val, cat_features=cat_cols)
final_model_cat_basic = cb.CatBoostRegressor(loss_function="RMSE", random_seed=random_state, train_dir="catboost_basic")
final_model_cat_basic.fit(train_pool, eval_set=val_pool, verbose=True)



############################################## Evaluation ############################################################
def evaluate_model(model, X, y, name):
    predictions = model.predict(X)
    rmse = root_mean_squared_error(y, predictions)
    print(f"{name} Performance:")
    print(f"Root Mean Squared Error: {rmse:.4f}")



evaluate_model(ols_model, X_val_ols, y_val, "OLS Regression")
evaluate_model(glm_lr_model, X_val_ols, y_val, "GLM (Gaussian)")
evaluate_model(con_ols_model, X_val_ols, y_val, "Constrained GLM (Gaussian)")

"""
Summary:
using Gaussian for count data (call_counts) is not ideal
"""
evaluate_model(final_model_ridge, X_val_reg, y_val, "Ridge Regression")
evaluate_model(final_model_lasso, X_val_reg, y_val, "Lasso Regression")

"""
Summary:
Regularized regression models still use assume Gaussian distribution for the target variable
using Gaussian for count data (call_counts) is not ideal
"""

evaluate_model(final_model_dt, X_val_tree, y_val, "Decision Tree Regressor")
evaluate_model(final_model_rf, X_val_tree, y_val, "Random Forest Regressor")
evaluate_model(final_model_xgb, X_val_tree, y_val, "XGBoost Regressor")


evaluate_model(final_model_lgbm, X_val_cat, y_val, "LightGBM Regressor")
evaluate_model(final_model_cat_basic, X_val_cat, y_val, "CatBoost Regressor")
