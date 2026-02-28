# =============================================================================
# AI-Driven Crop Ranking & Soil Health Recommendation System
# Pan-India Hackathon — Application & UI Stack
# =============================================================================
#
# FOLDER STRUCTURE (Future Modularization):
# ─────────────────────────────────────────
# /components        → Reusable UI widgets (metric cards, input groups, map)
# /pages             → Page-level views (home, recommender, insights)
# /services          → ML model wrappers, data fetchers, API clients
# /styles            → CSS injection helpers, theme tokens
# app.py             → Main entry — navigation, state init, page routing
#
# ─────────────────────────────────────────
# SESSION STATE PRIMER (for React / frontend engineers):
# ─────────────────────────────────────────
# Streamlit reruns the ENTIRE script from top to bottom on every interaction
# (button click, slider drag, page switch). This is fundamentally different
# from React, where only the affected component re-renders.
#
# st.session_state ≈ React's useState(), but GLOBAL across the app.
#   - In React:   const [n, setN] = useState(90)
#   - Streamlit:  if "n" not in st.session_state: st.session_state["n"] = 90
#
# Because the script reruns top-down, we initialise defaults at the TOP,
# then read / write them anywhere. This prevents inputs from resetting when
# the user switches sidebar pages.
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
# PHASE 7 — CUSTOM CSS INJECTION
# =============================================================================
# Injected early so every page benefits. Uses Google Fonts (Inter) and
# overrides default Streamlit padding / styling for a modern, brutalist,
# startup-grade aesthetic.
# =============================================================================

CUSTOM_CSS = """
<style>
    /* ── Google Font ─────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Global resets ───────────────────────────────────────────── */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif !important;
    }
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1rem !important;
        max-width: 1200px !important;
    }

    /* ── Sidebar styling ─────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1923 0%, #1a2634 100%);
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #e2e8f0 !important;
        font-weight: 500;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
    }

    /* ── Metric cards ────────────────────────────────────────────── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(99, 255, 182, 0.15);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #63ffb6 !important;
        font-weight: 700 !important;
    }

    /* ── Expander styling ────────────────────────────────────────── */
    details[data-testid="stExpander"] {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 10px;
    }
    details[data-testid="stExpander"] summary span {
        font-weight: 600 !important;
        color: #e2e8f0 !important;
        padding-left: 0.3rem;
    }

    /* ── Buttons ─────────────────────────────────────────────────── */
    .stButton>button {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        color: #fff !important;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        transition: all 0.2s ease;
        box-shadow: 0 4px 14px rgba(34, 197, 94, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(34, 197, 94, 0.45);
    }

    /* ── Hero heading gradient ───────────────────────────────────── */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #22c55e, #63ffb6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.15;
        margin-bottom: 0.4rem;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: #94a3b8;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    /* ── Feature card ────────────────────────────────────────────── */
    .feature-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(99, 255, 182, 0.1);
        border-radius: 14px;
        padding: 1.6rem;
        text-align: center;
        transition: all 0.25s ease;
        min-height: 180px;
    }
    .feature-card:hover {
        border-color: rgba(99, 255, 182, 0.35);
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
    }
    .feature-card h3 {
        color: #63ffb6 !important;
        font-size: 1.1rem;
        margin: 0.8rem 0 0.5rem 0;
    }
    .feature-card p {
        color: #94a3b8;
        font-size: 0.88rem;
        line-height: 1.5;
    }

    /* ── Section header ──────────────────────────────────────────── */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 0.3rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-sub {
        color: #64748b;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }

    /* ── Result card ─────────────────────────────────────────────── */
    .result-card {
        background: linear-gradient(135deg, #064e3b 0%, #0f2a1d 100%);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 14px;
        padding: 1.5rem;
    }
    .result-card h3 {
        color: #63ffb6 !important;
        margin-top: 0;
    }

    /* ── Table styling ───────────────────────────────────────────── */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    /* ── Hide default Streamlit menu & footer for clean demo ─────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Divider accent ──────────────────────────────────────────── */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #1e293b, transparent);
        margin: 1.5rem 0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# PHASE 3 — SESSION STATE INITIALISATION
# =============================================================================
# Think of this block as the "default props" or "initial useState values" that
# run once.  Because Streamlit reruns top-down on every interaction, we guard
# each key with `if key not in st.session_state` so existing values survive
# across reruns — exactly like how React preserves state between renders.
# =============================================================================

# Navigation state — equivalent to a React Router "location" in global state
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Home / Dashboard"

# ── Soil metric inputs (defaults are realistic Indian agriculture baselines) ──
if "input_nitrogen" not in st.session_state:
    st.session_state["input_nitrogen"] = 90       # kg/ha — typical for rice
if "input_phosphorus" not in st.session_state:
    st.session_state["input_phosphorus"] = 42      # kg/ha
if "input_potassium" not in st.session_state:
    st.session_state["input_potassium"] = 43       # kg/ha
if "input_ph" not in st.session_state:
    st.session_state["input_ph"] = 6.5             # mildly acidic — common

# ── Climate metric inputs ────────────────────────────────────────────────────
if "input_temperature" not in st.session_state:
    st.session_state["input_temperature"] = 25.0   # °C
if "input_humidity" not in st.session_state:
    st.session_state["input_humidity"] = 70.0       # %
if "input_rainfall" not in st.session_state:
    st.session_state["input_rainfall"] = 200.0      # mm

# ── Region selection ─────────────────────────────────────────────────────────
if "selected_region" not in st.session_state:
    st.session_state["selected_region"] = "Punjab"

# ── ML output cache — prevents re-computation on page switch ─────────────────
if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None


# =============================================================================
# PHASE 1 — SIDEBAR NAVIGATION
# =============================================================================
# The sidebar acts like a persistent nav bar.  We store the selected page in
# session_state so switching pages never resets user inputs (same idea as
# lifting state up in React so sibling components share data).
# =============================================================================

PAGES = ["Home / Dashboard", "Crop Recommender", "3D Data Insights"]

with st.sidebar:
    st.markdown("## 🌾 AgriRank AI")
    st.caption("Intelligent Crop Decisions for India")
    st.markdown("---")

    selected = st.radio(
        "Navigate",
        PAGES,
        index=PAGES.index(st.session_state["current_page"]),
        label_visibility="collapsed",
    )
    # Update state — this is like calling setCurrentPage(selected) in React.
    st.session_state["current_page"] = selected

    st.markdown("---")
    st.caption("Built for Pan-India Hackathon 2026")
    st.caption("v1.0 — Demo Build")


# =============================================================================
# STATIC DATA — Indian regions with coordinates, simulated yield & soil health
# =============================================================================
# This dataset drives both the Crop Recommender ranking table and the pydeck
# 3D map.  In production this would come from an API / database service.
# =============================================================================

REGIONS_DATA = [
    {"region": "Punjab",           "lat": 31.15,  "lon": 75.34, "yield": 4800, "soil_health": 82, "top_crop": "Wheat"},
    {"region": "Haryana",          "lat": 29.06,  "lon": 76.09, "yield": 4500, "soil_health": 78, "top_crop": "Wheat"},
    {"region": "Uttar Pradesh",    "lat": 26.85,  "lon": 80.91, "yield": 4200, "soil_health": 71, "top_crop": "Rice"},
    {"region": "West Bengal",      "lat": 22.99,  "lon": 87.85, "yield": 3900, "soil_health": 74, "top_crop": "Rice"},
    {"region": "Madhya Pradesh",   "lat": 23.47,  "lon": 77.95, "yield": 3600, "soil_health": 68, "top_crop": "Soybean"},
    {"region": "Maharashtra",      "lat": 19.75,  "lon": 75.71, "yield": 3400, "soil_health": 65, "top_crop": "Cotton"},
    {"region": "Rajasthan",        "lat": 27.02,  "lon": 74.22, "yield": 2800, "soil_health": 55, "top_crop": "Mustard"},
    {"region": "Tamil Nadu",       "lat": 11.13,  "lon": 78.66, "yield": 3700, "soil_health": 72, "top_crop": "Rice"},
    {"region": "Karnataka",        "lat": 15.32,  "lon": 75.71, "yield": 3500, "soil_health": 70, "top_crop": "Ragi"},
    {"region": "Andhra Pradesh",   "lat": 15.91,  "lon": 79.74, "yield": 3800, "soil_health": 73, "top_crop": "Rice"},
    {"region": "Gujarat",          "lat": 22.26,  "lon": 71.19, "yield": 3200, "soil_health": 62, "top_crop": "Groundnut"},
    {"region": "Bihar",            "lat": 25.10,  "lon": 85.31, "yield": 3100, "soil_health": 60, "top_crop": "Maize"},
    {"region": "Odisha",           "lat": 20.94,  "lon": 84.80, "yield": 2900, "soil_health": 58, "top_crop": "Rice"},
    {"region": "Assam",            "lat": 26.20,  "lon": 92.94, "yield": 2700, "soil_health": 64, "top_crop": "Tea"},
    {"region": "Kerala",           "lat": 10.85,  "lon": 76.27, "yield": 3000, "soil_health": 76, "top_crop": "Coconut"},
    {"region": "Telangana",        "lat": 18.11,  "lon": 79.02, "yield": 3650, "soil_health": 69, "top_crop": "Cotton"},
    {"region": "Chhattisgarh",     "lat": 21.27,  "lon": 81.87, "yield": 2600, "soil_health": 57, "top_crop": "Rice"},
    {"region": "Jharkhand",        "lat": 23.61,  "lon": 85.28, "yield": 2500, "soil_health": 54, "top_crop": "Rice"},
    {"region": "Uttarakhand",      "lat": 30.07,  "lon": 79.49, "yield": 2400, "soil_health": 66, "top_crop": "Rice"},
    {"region": "Himachal Pradesh", "lat": 31.10,  "lon": 77.17, "yield": 2200, "soil_health": 70, "top_crop": "Apple"},
]

REGION_NAMES = [r["region"] for r in REGIONS_DATA]

# Standard / ideal nutrient values for soil health comparison
IDEAL_NUTRIENTS = {"N": 100, "P": 50, "K": 50, "pH": 6.5}


# =============================================================================
# PHASE 4 — MOCKED ML PREDICTION
# =============================================================================
# This function simulates an ML model call.  In production, replace the body
# with a real model inference (e.g., scikit-learn pipeline, TFServing, or an
# API call to a hosted model).
#
# Signature mirrors what a real predict function would accept — a flat dict
# of feature values.
# =============================================================================

CROP_DATABASE = [
    {"name": "Rice",       "emoji": "🌾", "optimal_temp": (20, 35), "optimal_rain": (150, 300), "optimal_ph": (5.5, 7.0)},
    {"name": "Wheat",      "emoji": "🌿", "optimal_temp": (12, 25), "optimal_rain": (50, 150),  "optimal_ph": (6.0, 7.5)},
    {"name": "Maize",      "emoji": "🌽", "optimal_temp": (18, 32), "optimal_rain": (80, 200),  "optimal_ph": (5.5, 7.5)},
    {"name": "Cotton",     "emoji": "🏵️", "optimal_temp": (25, 40), "optimal_rain": (60, 150),  "optimal_ph": (6.0, 8.0)},
    {"name": "Sugarcane",  "emoji": "🎋", "optimal_temp": (20, 35), "optimal_rain": (150, 250), "optimal_ph": (6.0, 7.5)},
    {"name": "Soybean",    "emoji": "🫘", "optimal_temp": (20, 30), "optimal_rain": (60, 200),  "optimal_ph": (6.0, 7.0)},
    {"name": "Mustard",    "emoji": "🌼", "optimal_temp": (10, 25), "optimal_rain": (30, 80),   "optimal_ph": (6.0, 7.5)},
    {"name": "Groundnut",  "emoji": "🥜", "optimal_temp": (25, 35), "optimal_rain": (50, 130),  "optimal_ph": (6.0, 7.0)},
    {"name": "Millet",     "emoji": "🌿", "optimal_temp": (25, 35), "optimal_rain": (30, 100),  "optimal_ph": (5.5, 7.0)},
    {"name": "Lentil",     "emoji": "🟤", "optimal_temp": (15, 25), "optimal_rain": (30, 80),   "optimal_ph": (6.0, 7.5)},
]


def predict_crop(inputs_dict: dict) -> dict:
    """
    Mocked ML prediction function.

    Accepts:
        inputs_dict — {"N": int, "P": int, "K": int, "pH": float,
                        "temperature": float, "humidity": float,
                        "rainfall": float, "region": str}

    Returns:
        {"crop": str, "confidence": float, "yield_score": float,
         "runner_ups": list[dict], "suitability_notes": list[str]}
    """
    # ── Rule-based agro-climatic suitability scoring ─────────────────────────
    temp = inputs_dict["temperature"]
    rain = inputs_dict["rainfall"]
    ph   = inputs_dict["pH"]

    scored_crops = []
    for crop in CROP_DATABASE:
        score = 0.0
        # Temperature fit (0-35 pts)
        t_lo, t_hi = crop["optimal_temp"]
        if t_lo <= temp <= t_hi:
            score += 35
        else:
            score += max(0, 35 - abs(temp - (t_lo + t_hi) / 2) * 2)
        # Rainfall fit (0-35 pts)
        r_lo, r_hi = crop["optimal_rain"]
        if r_lo <= rain <= r_hi:
            score += 35
        else:
            score += max(0, 35 - abs(rain - (r_lo + r_hi) / 2) * 0.2)
        # pH fit (0-20 pts)
        p_lo, p_hi = crop["optimal_ph"]
        if p_lo <= ph <= p_hi:
            score += 20
        else:
            score += max(0, 20 - abs(ph - (p_lo + p_hi) / 2) * 8)
        # Nutrient bonus (0-10 pts)
        n_score = min(inputs_dict["N"] / 120, 1.0) * 4
        p_score = min(inputs_dict["P"] / 60, 1.0) * 3
        k_score = min(inputs_dict["K"] / 60, 1.0) * 3
        score += n_score + p_score + k_score

        scored_crops.append({**crop, "score": round(min(score, 100), 1)})

    scored_crops.sort(key=lambda c: c["score"], reverse=True)
    best = scored_crops[0]

    # Suitability notes
    notes = []
    if inputs_dict["N"] < 60:
        notes.append("⚠️ Nitrogen is low — consider urea or organic compost.")
    if inputs_dict["P"] < 30:
        notes.append("⚠️ Phosphorus is low — apply DAP or bone meal.")
    if inputs_dict["K"] < 30:
        notes.append("⚠️ Potassium is low — use MOP (Muriate of Potash).")
    if ph < 5.5:
        notes.append("⚠️ Soil is too acidic — apply lime to raise pH.")
    elif ph > 7.5:
        notes.append("⚠️ Soil is too alkaline — add gypsum or sulphur.")
    if not notes:
        notes.append("✅ Soil nutrients and pH are within healthy ranges.")

    return {
        "crop": best["name"],
        "emoji": best["emoji"],
        "confidence": best["score"],
        "yield_score": round(random.uniform(3.0, 5.5), 2),
        "runner_ups": [
            {"crop": c["name"], "emoji": c["emoji"], "score": c["score"]}
            for c in scored_crops[1:4]
        ],
        "suitability_notes": notes,
    }


# =============================================================================
# PAGE RENDERERS
# =============================================================================
# Each function below is a "page component".  The router at the bottom calls
# the appropriate one based on st.session_state["current_page"].
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — HOME / DASHBOARD PAGE
# ─────────────────────────────────────────────────────────────────────────────

def render_home():
    """Home page with hero section, 3D Spline embed, and feature cards."""

    # ── Hero Section ─────────────────────────────────────────────────────────
    st.markdown('<div class="hero-title">AgriRank AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">'
        'AI-Driven Crop Ranking &amp; Soil Health Recommendations for Indian Agriculture'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── 3D Spline Embed ─────────────────────────────────────────────────────
    # Using a public Spline scene for a modern 3D website feel.
    # Replace with your own Spline URL for a custom branded experience.
    # NOTE: Replace this URL with your own Spline scene for a branded experience.
    # The URL below is a public Spline community scene for demo purposes.
    st.components.v1.iframe(
        src="https://my.spline.design/worldplanet-a0694bdced0a4256ae56ee6a1c62993e/",
        height=420,
        scrolling=False,
    )

    st.markdown("---")

    # ── Feature Cards ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🚀 Platform Capabilities</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Everything you need for data-driven crop decisions</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="feature-card">'
            '<div style="font-size:2.2rem;">🧪</div>'
            '<h3>Soil Analysis</h3>'
            '<p>Input N-P-K, pH &amp; get actionable health diagnostics with improvement suggestions.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="feature-card">'
            '<div style="font-size:2.2rem;">🤖</div>'
            '<h3>AI Crop Ranking</h3>'
            '<p>ML-powered predictions rank the best-fit crops for your region and conditions.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="feature-card">'
            '<div style="font-size:2.2rem;">🗺️</div>'
            '<h3>Geo Insights</h3>'
            '<p>Interactive 3D India map to explore yield patterns &amp; regional soil health.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Quick stats strip ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 At a Glance</div>', unsafe_allow_html=True)
    qs1, qs2, qs3, qs4 = st.columns(4)
    qs1.metric("Regions Covered", "20", help="States & UTs with simulated data")
    qs2.metric("Crops in Database", "10", help="Rule-based suitability matching")
    qs3.metric("Avg Soil Health", "67%", help="Across all tracked regions")
    qs4.metric("Top Yield Region", "Punjab", help="4,800 kg/ha simulated")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 + 4 — CROP RECOMMENDER PAGE
# ─────────────────────────────────────────────────────────────────────────────

def render_crop_recommender():
    """
    Professional crop recommendation form with soil & climate inputs,
    ML prediction trigger, and rich result display.
    """

    st.markdown('<div class="section-header">🌱 Crop Recommender</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">'
        'Enter your soil &amp; climate parameters to get AI-powered crop recommendations'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Region Selector ──────────────────────────────────────────────────────
    st.session_state["selected_region"] = st.selectbox(
        "📍 Select Region",
        REGION_NAMES,
        index=REGION_NAMES.index(st.session_state["selected_region"]),
        help="Choose the Indian state / region you are farming in.",
    )

    # ── Two-column input layout ──────────────────────────────────────────────
    col_soil, col_climate = st.columns(2, gap="large")

    # ── Soil Metrics (left column) ───────────────────────────────────────────
    with col_soil:
        with st.expander("🧪 Soil Metrics", expanded=True):
            # Each input reads its default from session_state, and writes back
            # on change — like a controlled input in React.
            st.session_state["input_nitrogen"] = st.number_input(
                "Nitrogen (N) — kg/ha",
                min_value=0, max_value=300,
                value=st.session_state["input_nitrogen"],
                step=5,
                help="Typical range: 50-150 kg/ha. Rice needs ~90, Wheat ~120.",
            )
            st.session_state["input_phosphorus"] = st.number_input(
                "Phosphorus (P) — kg/ha",
                min_value=0, max_value=200,
                value=st.session_state["input_phosphorus"],
                step=5,
                help="Typical range: 20-80 kg/ha. Essential for root growth.",
            )
            st.session_state["input_potassium"] = st.number_input(
                "Potassium (K) — kg/ha",
                min_value=0, max_value=300,
                value=st.session_state["input_potassium"],
                step=5,
                help="Typical range: 20-80 kg/ha. Supports disease resistance.",
            )
            st.session_state["input_ph"] = st.slider(
                "Soil pH",
                min_value=3.5, max_value=9.5,
                value=st.session_state["input_ph"],
                step=0.1,
                help="6.0–7.0 is ideal for most crops. <5.5 = acidic, >7.5 = alkaline.",
            )

    # ── Climate Metrics (right column) ───────────────────────────────────────
    with col_climate:
        with st.expander("🌦️ Climate Metrics", expanded=True):
            st.session_state["input_temperature"] = st.number_input(
                "Temperature (°C)",
                min_value=0.0, max_value=55.0,
                value=st.session_state["input_temperature"],
                step=0.5,
                help="Annual avg for your region. India range: 10–45 °C.",
            )
            st.session_state["input_humidity"] = st.number_input(
                "Humidity (%)",
                min_value=10.0, max_value=100.0,
                value=st.session_state["input_humidity"],
                step=1.0,
                help="Relative humidity. Coastal regions: 70-90%, Arid: 20-40%.",
            )
            st.session_state["input_rainfall"] = st.number_input(
                "Rainfall (mm)",
                min_value=0.0, max_value=3000.0,
                value=st.session_state["input_rainfall"],
                step=10.0,
                help="Annual rainfall. Rajasthan ~300mm, Meghalaya ~2500mm.",
            )

    st.markdown("---")

    # ── Prediction Trigger ───────────────────────────────────────────────────
    if st.button("🔬  Run Crop Analysis", use_container_width=True):
        inputs = {
            "N": st.session_state["input_nitrogen"],
            "P": st.session_state["input_phosphorus"],
            "K": st.session_state["input_potassium"],
            "pH": st.session_state["input_ph"],
            "temperature": st.session_state["input_temperature"],
            "humidity": st.session_state["input_humidity"],
            "rainfall": st.session_state["input_rainfall"],
            "region": st.session_state["selected_region"],
        }

        # Simulate ML model call with spinner (Phase 4)
        with st.spinner("🧠 Running AI crop analysis..."):
            time.sleep(2)  # Simulated inference latency
            result = predict_crop(inputs)

        # Cache result in session_state so it persists across page switches
        # (like storing API response in React state / context)
        st.session_state["prediction_result"] = result

    # ── Results Display ──────────────────────────────────────────────────────
    result = st.session_state["prediction_result"]
    if result:
        st.markdown("---")
        st.markdown('<div class="section-header">📋 Analysis Results</div>', unsafe_allow_html=True)

        # Primary recommendation metrics
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Recommended Crop",
            f'{result["emoji"]} {result["crop"]}',
        )
        m2.metric(
            "Confidence Score",
            f'{result["confidence"]}%',
        )
        m3.metric(
            "Expected Yield",
            f'{result["yield_score"]} t/ha',
        )

        # Runner-up crops
        st.markdown("")
        with st.expander("🏆 Runner-Up Crops", expanded=True):
            ru_cols = st.columns(3)
            for i, ru in enumerate(result["runner_ups"]):
                with ru_cols[i]:
                    st.metric(
                        f"#{i+2} Pick",
                        f'{ru["emoji"]} {ru["crop"]}',
                        f'{ru["score"]}% fit',
                    )

        # Soil health comparison — current vs ideal
        with st.expander("🩺 Soil & Plant Health Diagnostics", expanded=True):
            diag_cols = st.columns(4)
            current_vals = {
                "N": st.session_state["input_nitrogen"],
                "P": st.session_state["input_phosphorus"],
                "K": st.session_state["input_potassium"],
                "pH": st.session_state["input_ph"],
            }
            for i, (nutrient, ideal) in enumerate(IDEAL_NUTRIENTS.items()):
                current = current_vals[nutrient]
                delta = round(current - ideal, 1)
                delta_str = f"{'+' if delta > 0 else ''}{delta}"
                diag_cols[i].metric(
                    f"{nutrient} (Ideal: {ideal})",
                    f"{current}",
                    delta_str,
                    delta_color="normal" if abs(delta) <= ideal * 0.2 else "inverse",
                )

            # Suitability notes
            st.markdown("**Improvement Guidance:**")
            for note in result["suitability_notes"]:
                st.markdown(f"- {note}")

        # Region ranking table
        with st.expander("📊 Top 10 Regions for Selected Crop", expanded=False):
            region_df = pd.DataFrame(REGIONS_DATA)
            # Sort by yield descending, take top 10
            region_df = region_df.sort_values("yield", ascending=False).head(10).reset_index(drop=True)
            region_df.index = region_df.index + 1  # 1-indexed rank
            region_df.columns = ["Region", "Latitude", "Longitude", "Yield (kg/ha)", "Soil Health (%)", "Top Crop"]
            st.dataframe(
                region_df[["Region", "Yield (kg/ha)", "Soil Health (%)", "Top Crop"]],
                use_container_width=True,
            )
            # Highlight selected region
            sel = st.session_state["selected_region"]
            match = region_df[region_df["Region"] == sel]
            if not match.empty:
                rank = match.index[0]
                st.info(f"📍 **{sel}** ranks **#{rank}** among the top regions.")
            else:
                st.warning(f"📍 **{sel}** is not in the top 10 by yield.")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6 — 3D DATA INSIGHTS PAGE (PYDECK MAP)
# ─────────────────────────────────────────────────────────────────────────────

def render_3d_insights():
    """
    Interactive 3D India map using pydeck ColumnLayer.  Displays simulated
    crop yield and soil health per region.
    """

    st.markdown('<div class="section-header">🗺️ 3D Data Insights — India</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">'
        'Explore crop yield &amp; soil health intensity across Indian states'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Map data prep ────────────────────────────────────────────────────────
    map_df = pd.DataFrame(REGIONS_DATA)

    # Colour by soil health — green gradient (higher = more vibrant green)
    map_df["color_r"] = (100 - map_df["soil_health"]).apply(lambda x: int(min(255, x * 3)))
    map_df["color_g"] = map_df["soil_health"].apply(lambda x: int(min(255, x * 2.8)))
    map_df["color_b"] = 80

    # ── Pydeck Layer ─────────────────────────────────────────────────────────
    column_layer = pdk.Layer(
        "ColumnLayer",
        data=map_df,
        get_position=["lon", "lat"],
        get_elevation="yield",
        elevation_scale=80,
        radius=35000,
        get_fill_color=["color_r", "color_g", "color_b", 200],
        pickable=True,
        auto_highlight=True,
    )

    # India-centric view
    view_state = pdk.ViewState(
        latitude=22.5,
        longitude=79.5,
        zoom=4.2,
        pitch=45,
        bearing=-15,
    )

    tooltip = {
        "html": (
            "<div style='font-family:Inter,sans-serif; padding:8px;'>"
            "<b style='font-size:14px; color:#63ffb6;'>{region}</b><br/>"
            "🌾 Yield: <b>{yield} kg/ha</b><br/>"
            "🩺 Soil Health: <b>{soil_health}%</b><br/>"
            "🏆 Top Crop: <b>{top_crop}</b>"
            "</div>"
        ),
        "style": {
            "backgroundColor": "#1e293b",
            "color": "#e2e8f0",
            "border": "1px solid rgba(99,255,182,0.3)",
            "border-radius": "8px",
        },
    }

    st.pydeck_chart(
        pdk.Deck(
            layers=[column_layer],
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style="mapbox://styles/mapbox/dark-v11",
        ),
        use_container_width=True,
    )

    st.markdown("---")

    # ── Top 5 / Top 10 Region Rankings ───────────────────────────────────────
    st.markdown('<div class="section-header">🏅 Regional Rankings</div>', unsafe_allow_html=True)

    rank_col1, rank_col2 = st.columns(2)

    with rank_col1:
        st.markdown("**Top 5 by Crop Yield**")
        top_yield = (
            map_df[["region", "yield", "top_crop"]]
            .sort_values("yield", ascending=False)
            .head(5)
            .reset_index(drop=True)
        )
        top_yield.index = top_yield.index + 1
        top_yield.columns = ["Region", "Yield (kg/ha)", "Top Crop"]
        st.dataframe(top_yield, use_container_width=True)

    with rank_col2:
        st.markdown("**Top 5 by Soil Health**")
        top_soil = (
            map_df[["region", "soil_health", "top_crop"]]
            .sort_values("soil_health", ascending=False)
            .head(5)
            .reset_index(drop=True)
        )
        top_soil.index = top_soil.index + 1
        top_soil.columns = ["Region", "Soil Health (%)", "Top Crop"]
        st.dataframe(top_soil, use_container_width=True)

    # ── Region detail selector ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🔍 Region Deep Dive</div>', unsafe_allow_html=True)

    detail_region = st.selectbox(
        "Select a region to inspect",
        REGION_NAMES,
        index=REGION_NAMES.index(st.session_state["selected_region"]),
        key="insights_region_select",
    )

    region_info = next(r for r in REGIONS_DATA if r["region"] == detail_region)

    dc1, dc2, dc3, dc4 = st.columns(4)
    dc1.metric("Region", region_info["region"])
    dc2.metric("Crop Yield", f'{region_info["yield"]} kg/ha')
    dc3.metric("Soil Health", f'{region_info["soil_health"]}%')
    dc4.metric("Top Crop", region_info["top_crop"])

    # Suitability bar (simple visual)
    suitability = region_info["soil_health"]
    if suitability >= 75:
        st.success(f"🟢 **{detail_region}** has excellent agricultural suitability ({suitability}%).")
    elif suitability >= 60:
        st.info(f"🟡 **{detail_region}** has moderate suitability ({suitability}%). Soil improvement recommended.")
    else:
        st.warning(f"🟠 **{detail_region}** has low suitability ({suitability}%). Significant soil enrichment needed.")


# =============================================================================
# PAGE ROUTER
# =============================================================================
# Simple conditional routing based on session state — equivalent to a React
# Router's <Routes> / <Route> pattern, but implemented as an if/elif chain
# because Streamlit doesn't have a component tree.
# =============================================================================

page = st.session_state["current_page"]

if page == "Home / Dashboard":
    render_home()
elif page == "Crop Recommender":
    render_crop_recommender()
elif page == "3D Data Insights":
    render_3d_insights()
