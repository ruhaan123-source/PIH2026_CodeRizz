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

import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np
import time
import random

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be the very first Streamlit command
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriRank AI — Crop Ranking & Soil Health",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

CUSTOM_CSS = """
<style>
    /* ── Google Fonts ─────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Global resets ───────────────────────────────────────── */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }

    /* ── Sidebar ─────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
    }

    /* ── Metric cards ────────────────────────────────────────── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(99, 255, 182, 0.15);
        border-radius: 12px;
        padding: 1rem 1.2rem;
    }
    div[data-testid="stMetric"] label { color: #94a3b8 !important; font-size: 0.8rem !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #63ffb6 !important; font-weight: 700 !important; }

    /* ── Buttons ─────────────────────────────────────────────── */
    .stButton>button {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        color: #fff !important;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(34, 197, 94, 0.4);
    }

    /* ── Section headers ─────────────────────────────────────── */
    .hero-title {
        font-size: 3rem; font-weight: 800;
        background: linear-gradient(135deg, #22c55e, #06b6d4);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle { color: #94a3b8; font-weight: 400; margin-bottom: 2rem; }
    .section-header {
        font-size: 1.8rem; font-weight: 700;
        background: linear-gradient(135deg, #22c55e, #06b6d4);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .section-sub { color: #94a3b8; font-size: 0.9rem; margin-bottom: 1.5rem; }

    /* ── Feature / info cards ────────────────────────────────── */
    .feature-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(99, 255, 182, 0.1);
        border-radius: 14px;
        padding: 1.6rem; text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(99, 255, 182, 0.08);
    }
    .feature-card h3 { color: #e2e8f0 !important; font-size: 1.1rem; margin: 0.8rem 0 0.4rem; }
    .feature-card p { color: #94a3b8; font-size: 0.85rem; }

    /* ── Styled section label ────────────────────────────────── */
    .styled-label {
        background: #111827; border: 1px solid #1e293b;
        border-radius: 10px; padding: 0.8rem 1.25rem 0.3rem;
        margin-bottom: 0.5rem;
    }
    .styled-label div {
        font-weight: 600; color: #e2e8f0; font-size: 0.95rem;
    }

    /* ── Login card ──────────────────────────────────────────── */
    .login-card {
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(99, 255, 182, 0.15);
        border-radius: 20px;
        padding: 2.5rem;
        max-width: 420px;
        margin: 0 auto;
    }

    /* ── Divider ─────────────────────────────────────────────── */
    hr {
        border: none; height: 1px;
        background: linear-gradient(90deg, transparent, #1e293b, transparent);
        margin: 1.5rem 0;
    }

    /* ── Hide Streamlit menu & footer ────────────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# DATA — Regions, Crops, Market Prices, Fertilizers
# =============================================================================

REGIONS_DATA = [
    {"region": "Punjab",           "lat": 31.15, "lon": 75.34, "yield": 4800, "soil_health": 82, "top_crop": "Wheat",
     "avg_temp": 24.5, "humidity": 55, "rainfall": 650, "water_ph": 7.2},
    {"region": "Haryana",          "lat": 29.06, "lon": 76.09, "yield": 4500, "soil_health": 78, "top_crop": "Wheat",
     "avg_temp": 25.0, "humidity": 50, "rainfall": 550, "water_ph": 7.5},
    {"region": "Uttar Pradesh",    "lat": 26.85, "lon": 80.91, "yield": 4200, "soil_health": 72, "top_crop": "Rice",
     "avg_temp": 26.0, "humidity": 65, "rainfall": 1000, "water_ph": 7.0},
    {"region": "West Bengal",      "lat": 22.99, "lon": 87.75, "yield": 3900, "soil_health": 74, "top_crop": "Rice",
     "avg_temp": 27.0, "humidity": 78, "rainfall": 1600, "water_ph": 6.8},
    {"region": "Andhra Pradesh",   "lat": 15.91, "lon": 79.74, "yield": 3800, "soil_health": 73, "top_crop": "Rice",
     "avg_temp": 28.5, "humidity": 72, "rainfall": 900, "water_ph": 7.1},
    {"region": "Madhya Pradesh",   "lat": 23.47, "lon": 77.95, "yield": 3600, "soil_health": 68, "top_crop": "Soybean",
     "avg_temp": 25.5, "humidity": 55, "rainfall": 1150, "water_ph": 7.3},
    {"region": "Maharashtra",      "lat": 19.75, "lon": 75.71, "yield": 3500, "soil_health": 65, "top_crop": "Cotton",
     "avg_temp": 27.0, "humidity": 60, "rainfall": 1100, "water_ph": 7.4},
    {"region": "Karnataka",        "lat": 15.32, "lon": 75.71, "yield": 3400, "soil_health": 67, "top_crop": "Sugarcane",
     "avg_temp": 26.5, "humidity": 65, "rainfall": 1350, "water_ph": 6.9},
    {"region": "Tamil Nadu",       "lat": 11.13, "lon": 78.66, "yield": 3300, "soil_health": 69, "top_crop": "Rice",
     "avg_temp": 28.0, "humidity": 70, "rainfall": 950, "water_ph": 7.0},
    {"region": "Kerala",           "lat": 10.85, "lon": 76.27, "yield": 3200, "soil_health": 76, "top_crop": "Coconut",
     "avg_temp": 27.5, "humidity": 80, "rainfall": 3000, "water_ph": 6.5},
    {"region": "Gujarat",          "lat": 22.26, "lon": 71.19, "yield": 3200, "soil_health": 62, "top_crop": "Groundnut",
     "avg_temp": 27.5, "humidity": 50, "rainfall": 800, "water_ph": 7.6},
    {"region": "Bihar",            "lat": 25.10, "lon": 85.31, "yield": 3100, "soil_health": 60, "top_crop": "Maize",
     "avg_temp": 26.0, "humidity": 68, "rainfall": 1200, "water_ph": 7.1},
    {"region": "Odisha",           "lat": 20.94, "lon": 84.80, "yield": 2900, "soil_health": 58, "top_crop": "Rice",
     "avg_temp": 27.0, "humidity": 72, "rainfall": 1500, "water_ph": 6.8},
    {"region": "Assam",            "lat": 26.20, "lon": 92.94, "yield": 2700, "soil_health": 64, "top_crop": "Tea",
     "avg_temp": 24.0, "humidity": 82, "rainfall": 2800, "water_ph": 6.3},
    {"region": "Rajasthan",        "lat": 27.02, "lon": 74.22, "yield": 2600, "soil_health": 52, "top_crop": "Millet",
     "avg_temp": 28.0, "humidity": 35, "rainfall": 350, "water_ph": 8.0},
    {"region": "Telangana",        "lat": 18.11, "lon": 79.02, "yield": 3500, "soil_health": 70, "top_crop": "Cotton",
     "avg_temp": 28.0, "humidity": 60, "rainfall": 950, "water_ph": 7.2},
    {"region": "Chhattisgarh",     "lat": 21.27, "lon": 81.87, "yield": 2800, "soil_health": 56, "top_crop": "Rice",
     "avg_temp": 26.5, "humidity": 65, "rainfall": 1400, "water_ph": 6.9},
    {"region": "Jharkhand",        "lat": 23.61, "lon": 85.28, "yield": 2500, "soil_health": 54, "top_crop": "Rice",
     "avg_temp": 25.5, "humidity": 60, "rainfall": 1300, "water_ph": 7.0},
    {"region": "Uttarakhand",      "lat": 30.07, "lon": 79.49, "yield": 2400, "soil_health": 66, "top_crop": "Rice",
     "avg_temp": 18.0, "humidity": 55, "rainfall": 1500, "water_ph": 6.7},
    {"region": "Himachal Pradesh", "lat": 31.10, "lon": 77.17, "yield": 2200, "soil_health": 70, "top_crop": "Apple",
     "avg_temp": 15.0, "humidity": 60, "rainfall": 1200, "water_ph": 6.5},
]

REGION_NAMES = [r["region"] for r in REGIONS_DATA]

# ── Crop database with optimal ranges & market prices ────────────────────────
CROP_DATABASE = [
    {"name": "Rice",       "emoji": "🌾", "group": "Cereals",    "optimal_temp": (20, 35), "optimal_rain": (150, 300), "optimal_ph": (5.5, 7.0), "ideal_n": 120, "ideal_p": 60, "ideal_k": 40, "market_price": 2183},
    {"name": "Wheat",      "emoji": "🌿", "group": "Cereals",    "optimal_temp": (12, 25), "optimal_rain": (50, 120),  "optimal_ph": (6.0, 7.5), "ideal_n": 150, "ideal_p": 60, "ideal_k": 40, "market_price": 2275},
    {"name": "Maize",      "emoji": "🌽", "group": "Cereals",    "optimal_temp": (21, 30), "optimal_rain": (80, 200),  "optimal_ph": (5.5, 7.5), "ideal_n": 135, "ideal_p": 55, "ideal_k": 45, "market_price": 2090},
    {"name": "Sugarcane",  "emoji": "🎋", "group": "Cash Crops", "optimal_temp": (25, 38), "optimal_rain": (150, 300), "optimal_ph": (6.0, 7.5), "ideal_n": 150, "ideal_p": 80, "ideal_k": 80, "market_price": 315},
    {"name": "Cotton",     "emoji": "☁️", "group": "Cash Crops", "optimal_temp": (25, 35), "optimal_rain": (80, 150),  "optimal_ph": (6.0, 8.0), "ideal_n": 100, "ideal_p": 50, "ideal_k": 50, "market_price": 6620},
    {"name": "Soybean",    "emoji": "🫘", "group": "Oilseeds",   "optimal_temp": (20, 30), "optimal_rain": (60, 150),  "optimal_ph": (6.0, 7.0), "ideal_n": 30,  "ideal_p": 60, "ideal_k": 40, "market_price": 4600},
    {"name": "Groundnut",  "emoji": "🥜", "group": "Oilseeds",   "optimal_temp": (25, 35), "optimal_rain": (50, 120),  "optimal_ph": (5.5, 7.0), "ideal_n": 25,  "ideal_p": 50, "ideal_k": 45, "market_price": 5850},
    {"name": "Lentil",     "emoji": "🟤", "group": "Pulses",     "optimal_temp": (15, 25), "optimal_rain": (30, 80),   "optimal_ph": (6.0, 7.5), "ideal_n": 20,  "ideal_p": 45, "ideal_k": 20, "market_price": 6425},
    {"name": "Millet",     "emoji": "🌱", "group": "Cereals",    "optimal_temp": (25, 35), "optimal_rain": (30, 100),  "optimal_ph": (5.5, 7.0), "ideal_n": 80,  "ideal_p": 40, "ideal_k": 40, "market_price": 2500},
    {"name": "Coconut",    "emoji": "🥥", "group": "Cash Crops", "optimal_temp": (25, 32), "optimal_rain": (150, 300), "optimal_ph": (5.5, 7.0), "ideal_n": 50,  "ideal_p": 30, "ideal_k": 120,"market_price": 3200},
    {"name": "Tea",        "emoji": "🍵", "group": "Cash Crops", "optimal_temp": (18, 28), "optimal_rain": (200, 400), "optimal_ph": (4.5, 6.0), "ideal_n": 100, "ideal_p": 50, "ideal_k": 50, "market_price": 28000},
    {"name": "Apple",      "emoji": "🍎", "group": "Cash Crops", "optimal_temp": (10, 22), "optimal_rain": (100, 200), "optimal_ph": (5.5, 6.8), "ideal_n": 70,  "ideal_p": 35, "ideal_k": 70, "market_price": 7500},
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
