import pandas as pd

train_df = pd.read_excel('Pearl Challenge data with dictionary.xlsx', sheet_name='TrainData')
test_df = pd.read_excel('Pearl Challenge data with dictionary.xlsx', sheet_name='TestData')

from joblib import load
import lightgbm as lgb

lgb_best = load("best_lgbm_model.pkl")

preprocessor = lgb_best.named_steps['preprocessor']
lgb_best_model = lgb_best.named_steps['regressor']

from sklearn.model_selection import train_test_split

drop_cols = ['FarmerID', 'Zipcode', 'CITY', 'DISTRICT', 'VILLAGE', 'Location']
target = 'Target_Variable/Total Income'
X = train_df.drop(columns=[target] + drop_cols)
y = train_df[target]

X_test = test_df.drop(columns=[target] + drop_cols)

cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

X[num_cols] = X[num_cols].fillna(X[num_cols].mean())

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_test = test_df.drop(columns=[target] + drop_cols)

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