import pandas as pd
import numpy as np
import pickle
import sys
import os

# Configuration
def get_asset_path(sub_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    assets_path = os.path.join(base_path, 'assets', sub_path)
    if os.path.exists(assets_path):
        return assets_path
    return os.path.join(base_path, sub_path)

MODEL_PATH = get_asset_path("models/fertilizer_model.pkl")
ENCODERS_PATH = get_asset_path("models/fertilizer_encoders.pkl")

def predict_fertilizer(temp, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODERS_PATH):
        raise FileNotFoundError("Fertilizer model assets missing. Please run fertilizer_train.py first.")
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(ENCODERS_PATH, 'rb') as f:
        encoders = pickle.load(f)
    
    # 1. Encode Categorical Inputs
    try:
        soil_code = encoders['Soil Type'].transform([soil_type])[0]
        crop_code = encoders['Crop Type'].transform([crop_type])[0]
    except ValueError as e:
        # Get list of known types for better error message
        known_soils = encoders['Soil Type'].classes_.tolist()
        known_crops = encoders['Crop Type'].classes_.tolist()
        return f"Error: Invalid type. Known Soils: {known_soils} | Known Crops: {known_crops}"

    input_df = pd.DataFrame([[temp, humidity, moisture, soil_code, crop_code, nitrogen, potassium, phosphorous]], 
                            columns=['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous'])
    
    # 3. Predict
    pred_idx = model.predict(input_df)[0]
    fertilizer_name = encoders['Fertilizer Name'].inverse_transform([pred_idx])[0]
    
    return fertilizer_name

if __name__ == "__main__":
    if len(sys.argv) < 9:
        print("Usage: python predict_fertilizer.py <Temp> <Humidity> <Moisture> <SoilType> <CropType> <N> <K> <P>")
        print("Example: python predict_fertilizer.py 26 52 38 Sandy Maize 37 0 0")
        sys.exit(1)
        
    try:
        t, h, m = map(float, sys.argv[1:4])
        soil = sys.argv[4]
        crop = sys.argv[5]
        n, k, p = map(float, sys.argv[6:9])
        
        result = predict_fertilizer(t, h, m, soil, crop, n, k, p)
        
        if result.startswith("Error"):
            print(result)
        else:
            print(f"\n--- Fertilizer Recommendation ---")
            print(f"Recommended Fertilizer: {result}")
            
    except Exception as e:
        print(f"Error during prediction: {e}")
