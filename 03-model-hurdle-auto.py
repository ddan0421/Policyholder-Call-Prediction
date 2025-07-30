import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import models 
from sklearn.model_selection import GridSearchCV, KFold

random_state = 42

############################################## Hurdle Method: First model zero or nonzero (Classificatoin) Then model non-zero count ######################################################
X_train = pd.read_csv("data/model_data_auto/X_train.csv")
X_val = pd.read_csv("data/model_data_auto/X_val.csv")
y_train = pd.read_csv("data/model_data_auto/y_train.csv").squeeze()
y_val = pd.read_csv("data/model_data_auto/y_val.csv").squeeze()
X_test = pd.read_csv("data/model_data_auto/X_test.csv")



############# First model: Zero or non-zero (Classification) #############
"""
Models to explore:
- Logit 
- KNN 
- SVM
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Naive Bayes
"""
y_train_class = (y_train > 0).astype(int)
y_val_class = (y_val > 0).astype(int)


#----------------------------------------------------------------------------#
#                          Logistic Regression Model                         #
#----------------------------------------------------------------------------#

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
X_val_logit = sm.add_constant(X_val_encoded)
X_test = sm.add_constant(X_test_encoded)

vif_score, vif_features = models.select_features_by_vif(X_train)

class_model = models.sm_logit(X_train[vif_features], y_train_class, method="ncg")
print(class_model.summary())


#----------------------------------------------------------------------------#
#                                   KNN                                      #
#----------------------------------------------------------------------------#





#----------------------------------------------------------------------------#
#                                 Evlaution                                  #
#----------------------------------------------------------------------------#

models.evaluate_classification_model(model=class_model, 
                                     X_test=X_val_logit[vif_features],
                                     y_test=y_val_class,
                                     model_name="Logit", 
                                     regression_type="binary", 
                                     library="statsmodels")