import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Configuration
DATA_PATH = r"C:\Users\cyber\Downloads\Fertilizer_dataset\Fertilizer Prediction.csv"
MODEL_SAVE_PATH = r"C:\codingalright\genailearning\housingagain\modelfiles\fertilizer_model.pkl"
ENCODERS_SAVE_PATH = r"C:\codingalright\genailearning\housingagain\modelfiles\fertilizer_encoders.pkl"

def train_fertilizer_model():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return

    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    
    # Clean column names (remove trailing spaces like 'Humidity ')
    df.columns = [c.strip() for c in df.columns]
    
    # 2. Preprocessing
    # Categorical columns to encode: Soil Type, Crop Type
    cat_cols = ['Soil Type', 'Crop Type']
    label_encoders = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        
    # Target encoding: Fertilizer Name
    target_le = LabelEncoder()
    df['Fertilizer Name'] = target_le.fit_transform(df['Fertilizer Name'])
    label_encoders['Fertilizer Name'] = target_le
    
    # 3. Model Training
    X = df.drop('Fertilizer Name', axis=1)
    y = df['Fertilizer Name']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"Model trained successfully! Accuracy: {accuracy:.4f}")
    
    # 4. Save Assets
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(ENCODERS_SAVE_PATH, 'wb') as f:
        pickle.dump(label_encoders, f)
        
    print(f"Model and encoders saved to {os.path.dirname(MODEL_SAVE_PATH)}")

if __name__ == "__main__":
    train_fertilizer_model()
