# =============================================================================
# AI-Driven Crop Ranking & Soil Health Recommendation System
# Pan-India Hackathon — 4-Page Application
# =============================================================================
#
# PAGE STRUCTURE:
#   Page 1 — Login / Greeting (Globe animation + auth gate)
#   Page 2 — Region-wise Top 10 Crops (Map selection + market values)
#   Page 3 — Recommendations (Fertilizers, pesticides, pH guidance)
#   Page 4 — India Map Info (3D pydeck map + About Us)
#
# SESSION STATE ≈ React useState() — global across the app.
# =============================================================================

import warnings
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np
import time
import random
import os
import pickle

from crop_inference import predict_crop_recommendations
from predict_fertilizer import predict_fertilizer

st.set_page_config(
    page_title="AgriRank AI — Crop Ranking & Soil Health",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Helper for path resolution in bundled apps
def get_asset_path(sub_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    # Check if we are running in a bundle
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    
    # Try the assets folder first (bundle/organized structure)
    assets_path = os.path.join(base_path, 'assets', sub_path)
    if os.path.exists(assets_path):
        return assets_path
    
    # Fallback to current directory (dev structure)
    local_path = os.path.join(base_path, sub_path)
    if os.path.exists(local_path):
        return local_path
        
    # Final fallback to parent directory if in child folder
    parent_path = os.path.join(os.path.dirname(base_path), sub_path)
    return parent_path

HISTORICAL_DATA_PATH = get_asset_path("data/district_crop_master.csv")
COORDS_DATA_PATH = get_asset_path("data/district_coords.csv")
CSS_PATH = get_asset_path("css/style.css")

def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css(CSS_PATH)

def normalize_state(name):
    if not isinstance(name, str): return name
    mapping = {
        "Andaman And Nicobar Islands": "Andaman And Nicobar",
        "Dadra And Nagar Haveli": "Dadra & Nagar Haveli",
        "Daman And Diu": "Daman & Diu",
        "Jammu And Kashmir": "Jammu & Kashmir",
        "Delhi": "Nct Of Delhi"
    }
    name = name.title().strip()
    return mapping.get(name, name)

@st.cache_data
def load_all_data():
    # Load historical
    if not os.path.exists(HISTORICAL_DATA_PATH):
        st.error(f"Data file not found: {HISTORICAL_DATA_PATH}")
        hist_df = pd.DataFrame()
    else:
        hist_df = pd.read_csv(HISTORICAL_DATA_PATH)
        for col in ['State', 'District', 'Crop']:
            if col in hist_df.columns:
                hist_df[col] = hist_df[col].apply(normalize_state) if col == 'State' else hist_df[col].str.title().str.strip()
    
    # Load coords
    if not os.path.exists(COORDS_DATA_PATH):
        st.warning(f"Coords file not found: {COORDS_DATA_PATH}")
        coords_df = pd.DataFrame()
    else:
        try:
            coords_df = pd.read_csv(COORDS_DATA_PATH, encoding='latin-1')
            coords_df.columns = [c.lstrip('ÿ').strip() for c in coords_df.columns]
            
            if not coords_df.empty:
                coords_df['State'] = coords_df['State'].apply(normalize_state)
                coords_df['District'] = coords_df['District'].str.title().str.strip()
                
                coords_df = coords_df.groupby(['State', 'District']).agg({
                    'Latitude': 'mean',
                    'Longitude': 'mean'
                }).reset_index()
        except Exception as e:
            st.error(f"Error loading map coordinates: {e}")
            coords_df = pd.DataFrame(columns=['State', 'District', 'Latitude', 'Longitude'])
        
    # Final safety check for columns
    SOIL_TYPES = ['Black', 'Clayey', 'Loamy', 'Red', 'Sandy']
    for col in ['State', 'District', 'Latitude', 'Longitude']:
        if col not in coords_df.columns:
            coords_df[col] = None
            
    return hist_df, coords_df

HISTORICAL_DF, DISTRICT_COORDS_DF = load_all_data()

# Static fallback coordinates for states
STATE_COORDINATES = {
    "Andhra Pradesh": (15.91, 79.74), "Arunachal Pradesh": (28.21, 94.72), "Assam": (26.20, 92.94),
    "Bihar": (25.10, 85.31), "Chhattisgarh": (21.27, 81.87), "Goa": (15.30, 74.12),
    "Gujarat": (22.26, 71.19), "Haryana": (29.06, 76.09), "Himachal Pradesh": (31.10, 77.17),
    "Jharkhand": (23.61, 85.28), "Karnataka": (15.32, 75.71), "Kerala": (10.85, 76.27),
    "Madhya Pradesh": (23.47, 77.95), "Maharashtra": (19.75, 75.71), "Manipur": (24.66, 93.90),
    "Meghalaya": (25.47, 91.36), "Mizoram": (23.16, 92.93), "Nagaland": (26.15, 94.56),
    "Odisha": (20.94, 84.80), "Punjab": (31.15, 75.34), "Rajasthan": (27.02, 74.22),
    "Sikkim": (27.53, 88.51), "Tamil Nadu": (11.13, 78.66), "Telangana": (18.11, 79.02),
    "Tripura": (23.74, 91.74), "Uttar Pradesh": (26.85, 80.91), "Uttarakhand": (30.07, 79.49),
    "West Bengal": (22.99, 87.75)
}

if not HISTORICAL_DF.empty:
    STATE_NAMES = sorted(HISTORICAL_DF['State'].unique().tolist())
else:
    STATE_NAMES = []

def get_district_center(state, district):
    if not DISTRICT_COORDS_DF.empty:
        match = DISTRICT_COORDS_DF[
            (DISTRICT_COORDS_DF['State'] == state) & 
            (DISTRICT_COORDS_DF['District'] == district)
        ]
        if not match.empty:
            return (match.iloc[0]['Latitude'], match.iloc[0]['Longitude'])
    
    # Fallback to state static coordinates if district center not found
    return STATE_COORDINATES.get(state, (20.59, 78.96))

# ── Crop database with items relevant for UI display ────────────────────────
CROP_DATABASE = [
    {"name": "Rice",       "emoji": "🌾", "group": "Cereals",    "ideal_n": 120, "ideal_p": 60, "ideal_k": 40, "market_price": 2183},
    {"name": "Wheat",      "emoji": "🌿", "group": "Cereals",    "ideal_n": 150, "ideal_p": 60, "ideal_k": 40, "market_price": 2275},
    {"name": "Maize",      "emoji": "🌽", "group": "Cereals",    "ideal_n": 135, "ideal_p": 55, "ideal_k": 45, "market_price": 2090},
    {"name": "Sugarcane",  "emoji": "🎋", "group": "Cash Crops", "ideal_n": 150, "ideal_p": 80, "ideal_k": 80, "market_price": 315},
    {"name": "Cotton",     "emoji": "☁️", "group": "Cash Crops", "ideal_n": 100, "ideal_p": 50, "ideal_k": 50, "market_price": 6620},
    {"name": "Soybean",    "emoji": "🫘", "group": "Oilseeds",   "ideal_n": 30,  "ideal_p": 60, "ideal_k": 40, "market_price": 4600},
    {"name": "Groundnut",  "emoji": "🥜", "group": "Oilseeds",   "ideal_n": 25,  "ideal_p": 50, "ideal_k": 45, "market_price": 5850},
    {"name": "Lentil",     "emoji": "🟤", "group": "Pulses",     "ideal_n": 20,  "ideal_p": 45, "ideal_k": 20, "market_price": 6425},
    {"name": "Millet",     "emoji": "🌱", "group": "Cereals",    "ideal_n": 80,  "ideal_p": 40, "ideal_k": 40, "market_price": 2500},
    {"name": "Coconut",    "emoji": "🥥", "group": "Cash Crops", "ideal_n": 50,  "ideal_p": 30, "ideal_k": 120,"market_price": 3200},
    {"name": "Tea",        "emoji": "🍵", "group": "Cash Crops", "ideal_n": 100, "ideal_p": 50, "ideal_k": 50, "market_price": 28000},
    {"name": "Apple",      "emoji": "🍎", "group": "Cash Crops", "ideal_n": 70,  "ideal_p": 35, "ideal_k": 70, "market_price": 7500},
]

CROP_GROUPS = sorted(set(c["group"] for c in CROP_DATABASE))

# ── Fertilizer database ──────────────────────────────────────────────────────
FERTILIZER_DB = {
    "low_n":  {"fertilizer": "Urea (46-0-0)", "dosage": "130–170 kg/ha", "note": "Apply in 2–3 split doses. First basal, rest at tillering & panicle."},
    "high_n": {"fertilizer": "Reduce Urea", "dosage": "Cut by 30–40%", "note": "Excess N causes lodging & pest susceptibility. Consider neem-coated urea."},
    "low_p":  {"fertilizer": "DAP (18-46-0)", "dosage": "100–130 kg/ha", "note": "Apply full dose at sowing. P is immobile — band placement is ideal."},
    "high_p": {"fertilizer": "Reduce DAP / SSP", "dosage": "Cut by 25–35%", "note": "Excess P locks out Zinc. Add ZnSO4 if deficiency symptoms appear."},
    "low_k":  {"fertilizer": "MOP (0-0-60)", "dosage": "80–100 kg/ha", "note": "Apply 50% basal + 50% at flowering. Critical for fruit & grain filling."},
    "high_k": {"fertilizer": "Reduce MOP", "dosage": "Cut by 20–30%", "note": "Excess K interferes with Mg & Ca uptake."},
    "low_ph": {"fertilizer": "Agricultural Lime (CaCO3)", "dosage": "2–4 tonnes/ha", "note": "Apply 2–3 weeks before sowing. Acidic soil limits nutrient availability."},
    "high_ph":{"fertilizer": "Gypsum (CaSO4)", "dosage": "2–5 tonnes/ha", "note": "Reduces alkalinity. Add organic matter (FYM / compost) to buffer pH."},
}

PESTICIDE_DB = {
    "Cereals":    [{"pest": "Stem Borer", "product": "Chlorantraniliprole 0.4% GR", "dosage": "10 kg/ha"}, {"pest": "Brown Plant Hopper", "product": "Pymetrozine 50% WG", "dosage": "300 g/ha"}, {"pest": "Blast", "product": "Tricyclazole 75% WP", "dosage": "300 g/ha"}],
    "Pulses":     [{"pest": "Pod Borer", "product": "Emamectin Benzoate 5% SG", "dosage": "220 g/ha"}, {"pest": "Wilt", "product": "Carbendazim 50% WP", "dosage": "1 kg/ha"}, {"pest": "Aphids", "product": "Imidacloprid 17.8% SL", "dosage": "100 ml/ha"}],
    "Oilseeds":   [{"pest": "White Grub", "product": "Chlorpyrifos 20% EC", "dosage": "2.5 L/ha"}, {"pest": "Tikka Disease", "product": "Mancozeb 75% WP", "dosage": "2 kg/ha"}, {"pest": "Jassids", "product": "Thiamethoxam 25% WG", "dosage": "100 g/ha"}],
    "Cash Crops": [{"pest": "Bollworm", "product": "Flubendiamide 39.35% SC", "dosage": "150 ml/ha"}, {"pest": "RedRot", "product": "Carbendazim 50% WP", "dosage": "1 kg/ha"}, {"pest": "Mealybug", "product": "Profenophos 50% EC", "dosage": "1 L/ha"}],
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_crop_icon(crop_name):
    for c in CROP_DATABASE:
        if c["name"].lower() == crop_name.lower():
            return c["emoji"]
    return "🌱"

def get_crop_price(crop_name):
    for c in CROP_DATABASE:
        if c["name"].lower() == crop_name.lower():
            return c["market_price"]
    return random.randint(1500, 5000)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_page" not in st.session_state:
    st.session_state.current_page = 1
if "selected_state" not in st.session_state:
    st.session_state.selected_state = STATE_NAMES[0] if STATE_NAMES else "Punjab"
if "selected_district" not in st.session_state:
    st.session_state.selected_district = None

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>👨‍🌾 AgriRank AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 0.8rem; opacity: 0.7;'>v2.0 — Model-Driven Analytics</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.authenticated:
        # State Selection
        st.session_state.selected_state = st.selectbox("Select State", STATE_NAMES, index=STATE_NAMES.index(st.session_state.selected_state) if st.session_state.selected_state in STATE_NAMES else 0)
        
        # District Selection
        districts = sorted(HISTORICAL_DF[HISTORICAL_DF['State'] == st.session_state.selected_state]['District'].unique().tolist())
        st.session_state.selected_district = st.selectbox("Select District", districts)

        st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)
        
        menu_options = {
            "🏠 Dashboard": 1,
            "📈 Crop Ranking": 2,
            "🧪 Recommendations": 3,
            "🗺️ Regional Map": 4
        }
        
        for label, page_idx in menu_options.items():
            if st.button(label, use_container_width=True, key=f"nav_{page_idx}"):
                st.session_state.current_page = page_idx
                st.rerun()
        
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()

# =============================================================================
# PAGE 1: LOGIN / GREETING
# =============================================================================
if st.session_state.current_page == 1:
    if not st.session_state.authenticated:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<div class='login-card'>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center; color: #63ffb6;'>Welcome to AgriRank AI</h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; font-size: 0.9rem; color: #94a3b8;'>AI-Driven Precision Agriculture</p>", unsafe_allow_html=True)
            
            username = st.text_input("Username", "admin")
            password = st.text_input("Password", type="password")
            
            if st.button("Access Dashboard", use_container_width=True):
                st.session_state.authenticated = True
                st.session_state.current_page = 2
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h1 class='hero-title'>Hello, Explorer!</h1>", unsafe_allow_html=True)
        st.markdown(f"<p class='hero-subtitle'>Welcome back to the dashboard. Currently viewing data for {st.session_state.selected_district}, {st.session_state.selected_state}.</p>", unsafe_allow_html=True)
        
        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Yield", "4,210 kg/ha", "+5% vs Avg")
        m2.metric("Soil Health", "74 / 100", "Stable")
        m3.metric("Rainfall Forecast", "820 mm", "Normal")
        m4.metric("Market Sentiment", "Bullish", "↑ Price Trend")

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='feature-card'><h3>🔍 Model-Driven Insights</h3><p>Our Random Forest & Gradient Boosted models analyze historical data and current soil parameters to give you high-accuracy crop rankings and fertilizer suggestions.</p></div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='feature-card'><h3>🌾 Precision Agriculture</h3><p>Maximize your ROI with region-specific recommendations. Our database covers over 20+ crops across all Indian agro-climatic zones.</p></div>", unsafe_allow_html=True)

# =============================================================================
# PAGE 2: CROP RANKING (MODEL DRIVEN)
# =============================================================================
elif st.session_state.current_page == 2:
    st.markdown("<h2 class='section-header'>📈 Predicted Crop Performance</h2>", unsafe_allow_html=True)
    st.markdown(f"<p class='section-sub'>Showing top 10 recommended crops for <b>{st.session_state.selected_district}, {st.session_state.selected_state}</b> using AI Yield Prediction.</p>", unsafe_allow_html=True)
    
    with st.spinner("Analyzing regional data..."):
        try:
            results_df, s_name, d_name = predict_crop_recommendations(st.session_state.selected_state, st.session_state.selected_district)
            
            if isinstance(results_df, str):
                st.error(results_df)
            else:
                top_10 = results_df.head(10)
                
                # Display Top 3 in prominent cards
                cols = st.columns(3)
                for i, (_, row) in enumerate(top_10.head(3).iterrows()):
                    with cols[i]:
                        icon = get_crop_icon(row['Crop'])
                        price = get_crop_price(row['Crop'])
                        st.markdown(f"""
                        <div class='feature-card'>
                            <div style='font-size: 2.5rem;'>{icon}</div>
                            <h3>Rank #{i+1}: {row['Crop']}</h3>
                            <p><b>Pred. Yield:</b> {row['Predicted_Yield']} {row['Units']}</p>
                            <p><b>Market Value:</b> ₹{price}/quintal</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Show all 10 in a table
                st.table(top_10)
        except Exception as e:
            st.error(f"Error calling yield model: {e}")

# =============================================================================
# PAGE 3: RECOMMENDATIONS (FERTILIZER MODEL)
# =============================================================================
elif st.session_state.current_page == 3:
    st.markdown("<h2 class='section-header'>🧪 Soil Health & AI Recommendations</h2>", unsafe_allow_html=True)
    st.markdown("<p class='section-sub'>Enter your soil test results to get AI-powered fertilizer and crop-specific guidance.</p>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1.5])
    
    with col_l:
        st.markdown("<div class='styled-label'><div>🌱 Target Crop & Soil Parameters</div></div>", unsafe_allow_html=True)
        # All Crops from the expanded dataset
        CROP_TYPES = [
            'Arhar/Tur', 'Bajra', 'Banana', 'Black Pepper', 'Cashewnut', 'Coconut', 
            'Coriander', 'Dry Chillies', 'Garlic', 'Ginger', 'Gram', 'Groundnut', 
            'Guar Seed', 'Jowar', 'Maize', 'Onion', 'Potato', 'Ragi', 'Rice', 
            'Sesamum', 'Small Millets', 'Soyabean', 'Sugarcane', 'Sunflower', 
            'Urad', 'Wheat', 'Turmeric', 'Barley', 'Cardamom', 'Rapeseed &Mustard', 
            'Moong(Green Gram)', 'Masoor', 'Castor Seed', 'Tobacco', 'Arecanut', 
            'Sweet Potato', 'Tapioca', 'Horse-Gram', 'Safflower'
        ]
        target_crop = st.selectbox("Crop you plan to grow", CROP_TYPES)
        
        soil_type = st.selectbox("Soil Type", ["Sandy", "Loamy", "Black", "Red", "Clayey"])
        
        n = st.slider("Nitrogen (N) Content", 0, 150, 50)
        p = st.slider("Phosphorous (P) Content", 0, 150, 40)
        k = st.slider("Potassium (K) Content", 0, 150, 30)
        
        # New inputs for fertilizer model
        st.markdown("<hr>", unsafe_allow_html=True)
        temp = st.number_input("Temperature (°C)", 10.0, 50.0, 28.0)
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
        moisture = st.number_input("Soil Moisture (%)", 0.0, 100.0, 40.0)

        if st.button("🧪 Analyze Soil Health", use_container_width=True):
            with st.spinner("Predicting optimal fertilizer..."):
                try:
                    # Calibrate NPK: UI is 0-150, Model trained on 0-42 (approx)
                    # We scale down to ensure we don't push the model out of distribution
                    scale = 42.0 / 150.0
                    scaled_n = n * scale
                    scaled_k = k * scale
                    scaled_p = p * scale
                    
                    fert_res = predict_fertilizer(temp, humidity, moisture, soil_type, target_crop, scaled_n, scaled_k, scaled_p)
                    
                    if "Error" in fert_res:
                        st.error(fert_res)
                    else:
                        st.success(f"AI Recommendation: **{fert_res}**")
                        st.session_state.fert_recommendation = fert_res
                except Exception as e:
                    st.error(f"Fertilizer Model Error: {e}")

    with col_r:
        if "fert_recommendation" in st.session_state:
            st.markdown(f"""
            <div style='background: rgba(34, 197, 94, 0.1); border: 1px solid #22c55e; border-radius: 12px; padding: 1.5rem;'>
                <h3 style='color: #22c55e; margin-top: 0;'>Recommended Solution</h3>
                <p style='font-size: 1.2rem; font-weight: 700;'>Use {st.session_state.fert_recommendation}</p>
                <p style='font-size: 0.9rem; color: #94a3b8;'>Based on your specific NPK levels and environmental conditions (Temp: {temp}°C, Hum: {humidity}%), our model suggests this fertilizer for maximum yield efficiency.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Fill the parameters on the left and click 'Analyze' to see the recommendation.")

# =============================================================================
# PAGE 4: MAP & ABOUT
# =============================================================================
elif st.session_state.current_page == 4:
    st.markdown("<h2 class='section-header'>🗺️ Regional Agricultural Map</h2>", unsafe_allow_html=True)
    st.markdown(f"<p class='section-sub'>Geographical distribution of <b>{st.session_state.selected_state}</b> districts and projected growth.</p>", unsafe_allow_html=True)
    
    # filter district coordinates for the current state
    state_district_coords = DISTRICT_COORDS_DF[DISTRICT_COORDS_DF['State'] == st.session_state.selected_state].copy()
    
    # Merge with yield data if available
    if not HISTORICAL_DF.empty:
        state_yields = HISTORICAL_DF[HISTORICAL_DF['State'] == st.session_state.selected_state]
        if not state_yields.empty:
            avg_yields = state_yields.groupby('District')['Yield'].mean().reset_index()
            state_district_coords = state_district_coords.merge(avg_yields, on='District', how='left')
        else:
            state_district_coords['Yield'] = 0
    else:
        state_district_coords['Yield'] = 0

    if state_district_coords.empty:
        st.warning(f"No geographical data available for {st.session_state.selected_state} in our coordinate database.")
    elif state_district_coords['Yield'].sum() == 0 and st.session_state.selected_state == "Andaman And Nicobar":
        st.info("📍 **Note**: Coordinate data is available for Andaman and Nicobar, but agricultural yield records are currently not available for this region.")
    
    # Map Visualization Mode Toggle
    map_mode = st.radio("Map Mode", ["National Overview", "State Focus"], horizontal=True, label_visibility="collapsed")
    
    # Get center for initial view
    center_lat, center_lon = get_district_center(st.session_state.selected_state, st.session_state.selected_district)
    
    @st.cache_data
    def load_geojson():
        url = "https://raw.githubusercontent.com/india-in-data/india-states-2019/master/india_states.geojson"
        try:
            import json, requests
            response = requests.get(url)
            return response.json()
        except:
            return None

    india_geojson = load_geojson()
    
    with st.spinner("Preparing map layers..."):
        # Normalize current selection for lookup
        norm_state = normalize_state(st.session_state.selected_state)
        
        layers = []
        
        # 1. Base GeoJson Layer for "Actual Map" feel
        if india_geojson:
            layers.append(pdk.Layer(
                "GeoJsonLayer",
                india_geojson,
                opacity=0.2,
                stroked=True,
                filled=True,
                extruded=False,
                get_fill_color="[200, 200, 200]",
                get_line_color="[255, 255, 255]",
                line_width_min_pixels=1,
            ))

        if map_mode == "National Overview":
            # National Heatmap
            layers.append(pdk.Layer(
                "HeatmapLayer",
                data=DISTRICT_COORDS_DF,
                get_position=["Longitude", "Latitude"],
                aggregation=pdk.types.String("SUM"),
                opacity=0.8,
            ))
            view_lat, view_lon, zoom = 22.59, 78.96, 4
            tooltip = {"html": "<b>National Agricultural Density</b>", "style": {"backgroundColor": "#0f172a", "color": "white"}}
        else:
            # State Focus - 3D Columns
            state_data = DISTRICT_COORDS_DF[DISTRICT_COORDS_DF['State'] == norm_state].copy()
            if state_data.empty:
                st.warning(f"Coordinate data not found for {st.session_state.selected_state}.")
                merged_map_data = pd.DataFrame()
            else:
                merged_map_data = pd.merge(
                    state_data, 
                    HISTORICAL_DF[['State', 'District', 'Avg_Yield']].groupby(['State', 'District']).mean().reset_index(),
                    on=['State', 'District'],
                    how='left'
                ).fillna(0)
                
                layers.append(pdk.Layer(
                    "ColumnLayer",
                    data=merged_map_data,
                    get_position=["Longitude", "Latitude"],
                    get_elevation="Avg_Yield",
                    elevation_scale=150, 
                    radius=6000,
                    get_fill_color="[63, 255, 182, 180]", 
                    pickable=True,
                    auto_highlight=True,
                ))
            view_lat, view_lon, zoom = center_lat, center_lon, 6.5
            tooltip = {
                "html": "<b>District:</b> {District}<br/><b>Average Yield:</b> {Avg_Yield} kg/ha",
                "style": {"backgroundColor": "#0f172a", "color": "white"}
            }

        view_state = pdk.ViewState(
            latitude=view_lat,
            longitude=view_lon,
            zoom=zoom,
            pitch=45 if map_mode == "State Focus" else 0,
        )
        
        st.pydeck_chart(pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            map_style='mapbox://styles/mapbox/dark-v11',
            tooltip=tooltip
        ))
        
        with st.expander("🔍 See Map Diagnostics"):
            if map_mode == "National Overview":
                st.write(f"Total National Points: {len(DISTRICT_COORDS_DF)}")
            else:
                st.write(f"Points in {st.session_state.selected_state}: {len(merged_map_data) if not state_data.empty else 0}")
    
    st.markdown("---", unsafe_allow_html=True)
    st.markdown("### About AgriRank AI")
    st.write("AgriRank AI is an advanced precision agriculture platform designed to empower Indian farmers with data-driven decision making. By leveraging localized coordinates from `district_coords.csv` and historical data, we provide deep spatial insights that help maximize efficiency and sustainability.")

