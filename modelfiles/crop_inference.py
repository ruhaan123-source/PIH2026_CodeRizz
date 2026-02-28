import pandas as pd
import numpy as np
import pickle
import os

MODEL_PATH = r"C:\codingalright\genailearning\housingagain\crop_yield_model.pkl"
METADATA_PATH = r"C:\codingalright\genailearning\housingagain\encoding_maps.pkl"
CROP_DATA_PATH = r"C:\codingalright\genailearning\unique_crop_requirements.csv"
HISTORICAL_DATA_PATH = r"C:\codingalright\genailearning\district_crop_master.csv"

def load_assets():
    if not all(os.path.exists(p) for p in [MODEL_PATH, METADATA_PATH, CROP_DATA_PATH, HISTORICAL_DATA_PATH]):
        raise FileNotFoundError("Model assets missing correctly.")
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(METADATA_PATH, 'rb') as f:
        encoding_maps = pickle.load(f)
        
    crop_reqs = pd.read_csv(CROP_DATA_PATH)
    historical = pd.read_csv(HISTORICAL_DATA_PATH)
    
    crop_reqs['Crop'] = crop_reqs['Crop'].str.title().str.strip()
    
    units_map = crop_reqs.set_index('Crop')['Units'].to_dict()
    
    return model, encoding_maps, crop_reqs, units_map, historical

def predict_crop_recommendations(state_name, district_name=None):
    model, encoding_maps, crop_reqs, units_map, historical = load_assets()
    
    state_name = state_name.title().strip()
    if district_name:
        district_name = district_name.title().strip()

    if district_name:
        context_data = historical[(historical['State'] == state_name) & (historical['District'] == district_name)]
        if context_data.empty:
             return f"District '{district_name}' in '{state_name}' not found."
    else:
        context_data = historical[historical['State'] == state_name]
        if context_data.empty:
            return f"State '{state_name}' not found."
        district_name = context_data['District'].mode()[0]

    test_rows = []
    for _, row in crop_reqs.iterrows():
        new_row = row.copy()
        new_row['State'] = state_name
        new_row['District'] = district_name
        new_row['season'] = row['crop_Season']
        new_row['Crop'] = row['Crop']
        test_rows.append(new_row)
        
    predict_df = pd.DataFrame(test_rows)
    crop_names_list = predict_df['Crop'].values
    
    cat_cols = ['season', 'crop_Season', 'crop_Soil_Texture', 'crop_Irrigation_Type']
    
    global_mean = 1.0
    for col in ['State', 'District', 'Crop']:
        predict_df[col] = predict_df[col].map(encoding_maps[col]).fillna(global_mean)
    
    predict_df_encoded = pd.get_dummies(predict_df, columns=cat_cols)
    
    expected_cols = model.get_booster().feature_names
    final_df = pd.DataFrame(index=predict_df.index)
    for col in expected_cols:
        final_df[col] = predict_df_encoded[col] if col in predict_df_encoded.columns else 0
            
    log_preds = model.predict(final_df)
    preds = np.expm1(log_preds)
    preds = np.maximum(preds, 0)
    
    results = []
    for crop, pred in zip(crop_names_list, preds):
        results.append({
            'Crop': crop,
            'Predicted_Yield': pred,
            'Units': units_map.get(crop, 'Tons/Ha')
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='Predicted_Yield', ascending=False)
    
    df_results['Predicted_Yield'] = df_results['Predicted_Yield'].map('{:,.2f}'.format)
    
    return df_results, state_name, district_name

if __name__ == "__main__":
    import sys
    state = "Andhra Pradesh"
    district = "Anantapur"
    
    if len(sys.argv) > 1:
        state = sys.argv[1]
    if len(sys.argv) > 2:
        district = sys.argv[2]
        
    try:
        res, s, d = predict_crop_recommendations(state, district)
        if isinstance(res, str):
            print(res)
        else:
            print(f"\n--- Crop Recommendations for {d}, {s} ---")
            print(res.to_string(index=False))
    except Exception as e:
        print(f"Error during inference: {e}")
