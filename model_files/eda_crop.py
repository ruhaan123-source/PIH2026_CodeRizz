import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV

# 1. Load Data
data=pd.read_csv(r"C:\codingalright\genailearning\state_crop_averages.csv")

# 2. Preprocessing
# Columns to use for state-level historical context
historical_cols = ['Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']

# Use One-Hot Encoding for low-cardinality features
data_encoded = pd.get_dummies(data, columns=['Season', 'crop_Season', 'crop_Soil_Texture', 'crop_Irrigation_Type'], drop_first=True)

y = data_encoded["Avg_Yield"]
x = data_encoded.drop("Avg_Yield", axis=1)

# Split FIRST to prevent data leakage
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 3. Proper Target Encoding: Store mappings for consistent prediction
encoding_maps = {}
for col in ['State', 'Crop']:
    target_mean = y_train.groupby(x_train[col]).mean()
    encoding_maps[col] = target_mean
    
    x_train[col] = x_train[col].map(target_mean)
    x_test[col] = x_test[col].map(target_mean)
    
    global_mean = y_train.mean()
    x_test[col] = x_test[col].fillna(global_mean)


model = XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8)
model.fit(x_train, y_train)

print(f"Model trained. R2 Score: {r2_score(y_test, model.predict(x_test)):.4f}")

def recommend_crops(state_name):
    """Predicts yields for all available crops in a given state."""
    state_data = data[data['State'] == state_name]
    if state_data.empty:
        return f"State '{state_name}' not found."
    
   
    state_avg_historical = state_data[historical_cols].mean()
    
    unique_crop_reqs = data.drop_duplicates(subset=['Crop'])
    
    test_rows = []
    for _, row in unique_crop_reqs.iterrows():
        new_row = row.copy()
        new_row['State'] = state_name
        for col in historical_cols:
            new_row[col] = state_avg_historical[col]
        test_rows.append(new_row)
    
    predict_df = pd.DataFrame(test_rows)
    crop_names = predict_df['Crop'].values
    
    predict_df_encoded = pd.get_dummies(predict_df, columns=['Season', 'crop_Season', 'crop_Soil_Texture', 'crop_Irrigation_Type'], drop_first=True)
    
    for col in x_train.columns:
        if col not in predict_df_encoded.columns:
            predict_df_encoded[col] = 0
            
  
    predict_df_encoded['State'] = predict_df_encoded['State'].map(encoding_maps['State']).fillna(y_train.mean())
    predict_df_encoded['Crop'] = predict_df_encoded['Crop'].map(encoding_maps['Crop']).fillna(y_train.mean())
    
    predict_df_encoded = predict_df_encoded[x_train.columns]
    preds = model.predict(predict_df_encoded)
    preds = np.maximum(preds, 0)
    
    return pd.DataFrame({
        'Crop': crop_names,
        'Predicted_Yield': preds
    }).sort_values(by='Predicted_Yield', ascending=False)


print("\nYield Predictions for all available crops in 'Andhra Pradesh':")
recommendations = recommend_crops('Andhra Pradesh')
print(recommendations.head(10))


print("\nExporting model and metadata...")


with open('crop_yield_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('encoding_maps.pkl', 'wb') as f:
    pickle.dump(encoding_maps, f)


unique_crops = data.drop_duplicates(subset=['Crop']).copy()
cols_to_keep = ['Crop'] + [col for col in unique_crops.columns if col.startswith('crop_')]
unique_crops = unique_crops[cols_to_keep]
unique_crops.to_csv('unique_crop_requirements.csv', index=False)

print("Export complete:")
print("- crop_yield_model.pkl")
print("- encoding_maps.pkl")
print("- unique_crop_requirements.csv")
