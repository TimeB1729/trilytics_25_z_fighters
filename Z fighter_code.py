import numpy as np
import os

import pandas as pd
train_df = pd.read_excel('Pearl Challenge data with dictionary.xlsx', sheet_name='TrainData')
test_df = pd.read_excel('Pearl Challenge data with dictionary.xlsx', sheet_name='TestData')

drop_cols = ['FarmerID', 'Zipcode', 'CITY', 'DISTRICT', 'VILLAGE', 'Location']
target = 'Target_Variable/Total Income'
X = train_df.drop(columns=[target] + drop_cols)
y = train_df[target]

X_test = test_df.drop(columns=[target] + drop_cols)

cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessing data

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ]
)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Bayesian optimization for hyperparameter tuning in RandomizedSearchCV

import optuna
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from lightgbm import early_stopping

pipeline = Pipeline([
    ("preprocessor", preprocessor)
])

# Step 1: Define objective
def objective(trial):
    # Step 2: Suggest hyperparameters
    param = {
        'objective': 'regression',
        'metric': 'mape',
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'n_estimators': 1000,
        'verbosity': -1,
    }

    # Step 3: Train-validation split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Preprocess the data
    X_train_processed = pipeline.fit_transform(X_train)
    X_valid_processed = pipeline.transform(X_valid)

    # Step 5: Train LightGBM
    model = lgb.LGBMRegressor(**param)
    model.fit(
        X_train_processed, y_train,
        eval_set=[(X_valid_processed, y_valid)],
        eval_metric='mape',
        callbacks=[early_stopping(stopping_rounds=50)],
    )

    # Step 6: Evaluate
    preds = model.predict(X_valid_processed)
    mape = mean_absolute_percentage_error(y_valid, preds)
    return mape

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)  # ← Start with 50, go higher if needed

print(f"Validation MAPE: {study.best_value*100} %")
print("Best Hyperparameters:", study.best_params)

# Save the best model

lgb_best = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', lgb.LGBMRegressor(**study.best_params, n_estimators=1000))
])

lgb_best.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_percentage_error

y_pred = lgb_best.predict(X_val)
print(f"Validation MAPE: {mean_absolute_percentage_error(y_val, y_pred)*100} %")

# Predicting test data
predictions = lgb_best.predict(X_test)

output_df = pd.DataFrame({
    'FarmerID': test_df['FarmerID'],
    'Target_Variable/Total Income': predictions
})

# Convert FarmerID to string to preserve formatting
output_df['FarmerID'] = output_df['FarmerID'].astype(str)

# Save to CSV file
output_df.to_csv("Z fighter_predictions.csv", index=False)

print("Predictions saved to 'Z fighter_predictions.csv'")
