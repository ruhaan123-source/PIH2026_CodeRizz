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

    /* ── Hide sidebar collapse button (broken Material Icon text) ── */
    button[data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"],
    button[kind="headerNoPadding"],
    section[data-testid="stSidebar"] > div:first-child > button:first-child {
        display: none !important;
        visibility: hidden !important;
    }
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


# =============================================================================
# SESSION STATE INITIALISATION
# =============================================================================

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Region-wise Top Crops"
if "selected_region" not in st.session_state:
    st.session_state["selected_region"] = "Punjab"
if "input_nitrogen" not in st.session_state:
    st.session_state["input_nitrogen"] = 90
if "input_phosphorus" not in st.session_state:
    st.session_state["input_phosphorus"] = 42
if "input_potassium" not in st.session_state:
    st.session_state["input_potassium"] = 43
if "page2_results" not in st.session_state:
    st.session_state["page2_results"] = None
if "rec_nitrogen" not in st.session_state:
    st.session_state["rec_nitrogen"] = 90
if "rec_phosphorus" not in st.session_state:
    st.session_state["rec_phosphorus"] = 42
if "rec_potassium" not in st.session_state:
    st.session_state["rec_potassium"] = 43
if "rec_group" not in st.session_state:
    st.session_state["rec_group"] = "Cereals"
if "rec_crop" not in st.session_state:
    st.session_state["rec_crop"] = "Rice"
if "page3_results" not in st.session_state:
    st.session_state["page3_results"] = None


# =============================================================================
# HELPER — Crop suitability scoring
# =============================================================================

def score_crop(crop, n, p, k, temp, rain, ph):
    score = 0
    tl, th = crop["optimal_temp"]
    if tl <= temp <= th:
        score += 30
    else:
        score += max(0, 30 - abs(temp - (tl+th)/2) * 2)
    rl, rh = crop["optimal_rain"]
    if rl <= rain <= rh:
        score += 25
    else:
        score += max(0, 25 - abs(rain - (rl+rh)/2) / 10)
    pl, ph_h = crop["optimal_ph"]
    if pl <= ph <= ph_h:
        score += 25
    else:
        score += max(0, 25 - abs(ph - (pl+ph_h)/2) * 10)
    npk_diff = (abs(n - crop["ideal_n"]) + abs(p - crop["ideal_p"]) + abs(k - crop["ideal_k"])) / 3
    score += max(0, 20 - npk_diff / 5)
    return round(min(100, max(0, score)), 1)


GLOBE_HTML = """
<div style="position:relative;width:100%;height:400px;background:linear-gradient(135deg,#0a0f1a 0%,#0f1923 50%,#0a1628 100%);border-radius:16px;overflow:hidden;border:1px solid rgba(99,255,182,0.1);">
    <canvas id="globeCanvas" style="position:absolute;top:0;left:0;width:100%;height:100%;"></canvas>
    <div style="position:absolute;bottom:20px;left:50%;transform:translateX(-50%);text-align:center;z-index:2;">
        <div style="font-family:Inter,sans-serif;font-size:0.8rem;color:rgba(99,255,182,0.6);letter-spacing:0.15em;text-transform:uppercase;">
            Powered by AI &bull; 20 Regions &bull; 12 Crops
        </div>
    </div>
    <script>
    (function(){
        const c=document.getElementById('globeCanvas');const ctx=c.getContext('2d');let W,H;
        function resize(){W=c.width=c.offsetWidth;H=c.height=c.offsetHeight;}resize();window.addEventListener('resize',resize);
        const particles=[];for(let i=0;i<120;i++){particles.push({theta:Math.random()*Math.PI*2,phi:Math.acos(2*Math.random()-1),r:120+Math.random()*10,size:Math.random()*2+0.5,speed:0.002+Math.random()*0.003,glow:Math.random()>0.7});}
        const indiaPoints=[[28.6,77.2],[19.1,72.9],[13.1,80.3],[22.6,88.4],[26.9,81],[23.3,77.4],[21.2,79],[25,85.1],[15.5,74],[11.6,78.1],[17.4,78.5],[26.8,75.8],[31.1,77.2],[30.7,76.8],[22.3,71.2],[20.9,85.1],[24.8,93],[27.2,94.7]].map(([lat,lon])=>({theta:(90-lat)*Math.PI/180,phi:(lon-80)*Math.PI/180,r:120}));
        const rings=[];for(let i=0;i<3;i++){const pts=[];for(let j=0;j<60;j++){pts.push({a:j/60*Math.PI*2,tilt:0.3+i*0.25,rOff:140+i*25});}rings.push(pts);}
        let frame=0;
        function draw(){
            ctx.clearRect(0,0,W,H);const cx=W/2,cy=H/2;const rot=frame*0.008;frame++;
            const grd=ctx.createRadialGradient(cx,cy,30,cx,cy,200);grd.addColorStop(0,'rgba(34,197,94,0.06)');grd.addColorStop(1,'rgba(0,0,0,0)');ctx.fillStyle=grd;ctx.fillRect(0,0,W,H);
            ctx.strokeStyle='rgba(99,255,182,0.08)';ctx.lineWidth=0.5;for(let i=1;i<=5;i++){ctx.beginPath();ctx.arc(cx,cy,i*24,0,Math.PI*2);ctx.stroke();}
            for(let m=0;m<6;m++){ctx.beginPath();ctx.strokeStyle='rgba(99,255,182,0.05)';const a=m*Math.PI/6+rot;for(let t=0;t<=Math.PI;t+=0.05){const x=cx+120*Math.sin(t)*Math.cos(a);const y=cy-120*Math.cos(t);t===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}ctx.stroke();}
            particles.forEach(p=>{const t=p.theta+rot*p.speed*50;const x=cx+p.r*Math.sin(p.phi)*Math.cos(t);const y=cy-p.r*Math.cos(p.phi);const z=p.r*Math.sin(p.phi)*Math.sin(t);if(z<0)return;const alpha=0.3+0.7*(z/p.r);if(p.glow){ctx.shadowBlur=8;ctx.shadowColor='rgba(99,255,182,0.5)';}ctx.beginPath();ctx.arc(x,y,p.size*alpha,0,Math.PI*2);ctx.fillStyle=`rgba(99,255,182,${alpha*0.7})`;ctx.fill();ctx.shadowBlur=0;});
            indiaPoints.forEach(p=>{const t=p.phi+rot;const x=cx+p.r*Math.sin(p.theta)*Math.cos(t);const y=cy-p.r*Math.cos(p.theta);const z=p.r*Math.sin(p.theta)*Math.sin(t);if(z<0)return;const alpha=0.4+0.6*(z/p.r);ctx.shadowBlur=12;ctx.shadowColor='rgba(34,197,94,0.8)';ctx.beginPath();ctx.arc(x,y,3*alpha,0,Math.PI*2);ctx.fillStyle=`rgba(34,197,94,${alpha})`;ctx.fill();ctx.shadowBlur=0;});
            rings.forEach((pts,ri)=>{ctx.beginPath();ctx.strokeStyle=`rgba(6,182,212,${0.08+ri*0.03})`;ctx.lineWidth=0.6;pts.forEach((p,j)=>{const a=p.a+rot*0.5;const x=cx+p.rOff*Math.cos(a);const y=cy+p.rOff*Math.sin(a)*Math.sin(p.tilt);j===0?ctx.moveTo(x,y):ctx.lineTo(x,y);});ctx.closePath();ctx.stroke();const oa=rot*1.5+ri*2;const ox=cx+pts[0].rOff*Math.cos(oa);const oy=cy+pts[0].rOff*Math.sin(oa)*Math.sin(pts[0].tilt);ctx.beginPath();ctx.arc(ox,oy,2,0,Math.PI*2);ctx.fillStyle='rgba(6,182,212,0.7)';ctx.shadowBlur=6;ctx.shadowColor='rgba(6,182,212,0.6)';ctx.fill();ctx.shadowBlur=0;});
            requestAnimationFrame(draw);
        }
        draw();
    })();
    </script>
</div>
"""


# =============================================================================
# PAGE 1 — LOGIN / GREETING
# =============================================================================

def render_login():
    st.markdown('<div class="hero-title" style="text-align:center;">AgriRank AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle" style="text-align:center;">'
        'AI-Driven Crop Ranking &amp; Soil Health Recommendations for Indian Agriculture'
        '</div>', unsafe_allow_html=True,
    )
    st.components.v1.html(GLOBE_HTML, height=420)
    st.markdown("")
    col_l, col_c, col_r = st.columns([1, 1.2, 1])
    with col_c:
        st.markdown(
            '<div class="login-card">'
            '<h3 style="color:#63ffb6;text-align:center;margin-bottom:0.2rem;">🔐 Welcome Back</h3>'
            '<p style="color:#94a3b8;text-align:center;font-size:0.85rem;">Sign in to access the dashboard</p>'
            '</div>', unsafe_allow_html=True,
        )
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        if st.button("🚀  Login", use_container_width=True):
            if username and password:
                if username == "admin" and password == "admin123":
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Try admin / admin123")
            else:
                st.warning("Please enter both username and password.")
        st.markdown(
            '<div style="text-align:center;margin-top:1rem;color:#475569;font-size:0.75rem;">'
            'Demo: <b>admin</b> / <b>admin123</b> &nbsp;|&nbsp; Built for Pan-India Hackathon 2026'
            '</div>', unsafe_allow_html=True,
        )


# =============================================================================
# PAGE 2 — REGION-WISE TOP 10 CROPS
# =============================================================================

def render_region_crops():
    st.markdown('<div class="section-header">🌾 Region-wise Top Crops</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Select your region, enter soil nutrients, and discover the best crops with live market values</div>', unsafe_allow_html=True)

    map_df = pd.DataFrame(REGIONS_DATA)
    map_df["color_r"] = 34; map_df["color_g"] = 197; map_df["color_b"] = 94; map_df["color_a"] = 200
    scatter = pdk.Layer("ScatterplotLayer", data=map_df, get_position=["lon", "lat"], get_radius=40000,
        get_fill_color=["color_r", "color_g", "color_b", "color_a"], pickable=True, auto_highlight=True)
    view = pdk.ViewState(latitude=22.5, longitude=79.5, zoom=4.2, pitch=0)
    tooltip = {"html": "<div style='font-family:Inter;padding:6px;'><b style='color:#63ffb6;'>{region}</b><br/>🌡️ {avg_temp}°C &bull; 💧 {rainfall}mm</div>",
        "style": {"backgroundColor": "#1e293b", "color": "#e2e8f0", "border": "1px solid rgba(99,255,182,0.3)", "border-radius": "8px"}}
    st.pydeck_chart(pdk.Deck(layers=[scatter], initial_view_state=view, tooltip=tooltip,
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"), use_container_width=True)

    st.markdown("---")

    col_reg, col_n, col_p, col_k = st.columns([2, 1, 1, 1])
    with col_reg:
        st.session_state["selected_region"] = st.selectbox("📍 Select Region", REGION_NAMES,
            index=REGION_NAMES.index(st.session_state["selected_region"]), help="Choose from the map above or select here.")
    with col_n:
        st.session_state["input_nitrogen"] = st.number_input("Nitrogen (N)", min_value=0, max_value=300, value=st.session_state["input_nitrogen"], step=5)
    with col_p:
        st.session_state["input_phosphorus"] = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=st.session_state["input_phosphorus"], step=5)
    with col_k:
        st.session_state["input_potassium"] = st.number_input("Potassium (K)", min_value=0, max_value=300, value=st.session_state["input_potassium"], step=5)

    region_data = next(r for r in REGIONS_DATA if r["region"] == st.session_state["selected_region"])
    st.markdown('<div class="styled-label"><div>🌡️ Auto-Fetched Climate Data — ' + region_data["region"] + '</div></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temperature", f'{region_data["avg_temp"]} °C')
    c2.metric("Humidity", f'{region_data["humidity"]} %')
    c3.metric("Rainfall", f'{region_data["rainfall"]} mm')
    c4.metric("Water pH", f'{region_data["water_ph"]}')

    st.markdown("---")

    if st.button("🔬  Find Top 10 Crops", use_container_width=True):
        with st.spinner("Analyzing agro-climatic suitability..."):
            time.sleep(1.5)
            results = []
            for crop in CROP_DATABASE:
                s = score_crop(crop, st.session_state["input_nitrogen"], st.session_state["input_phosphorus"],
                    st.session_state["input_potassium"], region_data["avg_temp"], region_data["rainfall"], region_data["water_ph"])
                results.append({"Rank": 0, "Crop": f'{crop["emoji"]} {crop["name"]}', "Group": crop["group"],
                    "Suitability": f"{s}%", "Score": s, "Market Price (₹/q)": f'₹{crop["market_price"]:,}',
                    "Yield Potential": f'{random.randint(25, 55) / 10} t/ha'})
            results.sort(key=lambda x: x["Score"], reverse=True)
            for i, r in enumerate(results[:10]):
                r["Rank"] = i + 1
            st.session_state["page2_results"] = results[:10]

    if st.session_state["page2_results"]:
        st.markdown('<div class="styled-label"><div>🏆 Top 10 Recommended Crops for ' + st.session_state["selected_region"] + '</div></div>', unsafe_allow_html=True)
        top3 = st.session_state["page2_results"][:3]
        cols = st.columns(3)
        for i, crop in enumerate(top3):
            with cols[i]:
                st.metric(f"#{crop['Rank']} Recommended", crop["Crop"], f'{crop["Suitability"]} match • {crop["Market Price (₹/q)"]}')
        df = pd.DataFrame(st.session_state["page2_results"])
        df = df[["Rank", "Crop", "Group", "Suitability", "Market Price (₹/q)", "Yield Potential"]]
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.info("💡 Market prices are MSP / mandi averages (mocked). Integrate Agmarknet API for live data.")


# =============================================================================
# PAGE 3 — RECOMMENDATIONS
# =============================================================================

def render_recommendations():
    st.markdown('<div class="section-header">💊 Crop Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Enter your soil nutrients and crop to get precise fertilizer, pesticide, and pH recommendations</div>', unsafe_allow_html=True)

    col_grp, col_crop = st.columns(2)
    with col_grp:
        st.session_state["rec_group"] = st.selectbox("🏷️ Crop Group", CROP_GROUPS,
            index=CROP_GROUPS.index(st.session_state["rec_group"]), help="Category of the crop you are growing.")
    with col_crop:
        group_crops = [c["name"] for c in CROP_DATABASE if c["group"] == st.session_state["rec_group"]]
        if st.session_state["rec_crop"] not in group_crops:
            st.session_state["rec_crop"] = group_crops[0]
        st.session_state["rec_crop"] = st.selectbox("🌱 Select Crop", group_crops,
            index=group_crops.index(st.session_state["rec_crop"]), help="Specific crop you want recommendations for.")

    st.markdown('<div class="styled-label"><div>🧪 Your Soil Nutrients</div></div>', unsafe_allow_html=True)
    cn, cp, ck = st.columns(3)
    with cn:
        st.session_state["rec_nitrogen"] = st.number_input("Nitrogen (N) — kg/ha", min_value=0, max_value=300, value=st.session_state["rec_nitrogen"], step=5, key="rec_n_input")
    with cp:
        st.session_state["rec_phosphorus"] = st.number_input("Phosphorus (P) — kg/ha", min_value=0, max_value=200, value=st.session_state["rec_phosphorus"], step=5, key="rec_p_input")
    with ck:
        st.session_state["rec_potassium"] = st.number_input("Potassium (K) — kg/ha", min_value=0, max_value=300, value=st.session_state["rec_potassium"], step=5, key="rec_k_input")

    st.markdown("---")

    if st.button("🔍  Get Recommendations", use_container_width=True):
        with st.spinner("Analyzing soil health & generating recommendations..."):
            time.sleep(1.5)
            crop = next(c for c in CROP_DATABASE if c["name"] == st.session_state["rec_crop"])
            st.session_state["page3_results"] = {"crop": crop, "n": st.session_state["rec_nitrogen"], "p": st.session_state["rec_phosphorus"], "k": st.session_state["rec_potassium"]}

    res = st.session_state["page3_results"]
    if res:
        crop = res["crop"]
        n_val, p_val, k_val = res["n"], res["p"], res["k"]

        st.markdown('<div class="styled-label"><div>📊 Your Soil vs Ideal for ' + crop["emoji"] + ' ' + crop["name"] + '</div></div>', unsafe_allow_html=True)
        dc1, dc2, dc3 = st.columns(3)
        n_delta = n_val - crop["ideal_n"]
        p_delta = p_val - crop["ideal_p"]
        k_delta = k_val - crop["ideal_k"]
        dc1.metric(f"Nitrogen (Ideal: {crop['ideal_n']})", f"{n_val} kg/ha", f"{'+' if n_delta > 0 else ''}{n_delta}",
            delta_color="normal" if abs(n_delta) <= crop["ideal_n"] * 0.2 else "inverse")
        dc2.metric(f"Phosphorus (Ideal: {crop['ideal_p']})", f"{p_val} kg/ha", f"{'+' if p_delta > 0 else ''}{p_delta}",
            delta_color="normal" if abs(p_delta) <= crop["ideal_p"] * 0.2 else "inverse")
        dc3.metric(f"Potassium (Ideal: {crop['ideal_k']})", f"{k_val} kg/ha", f"{'+' if k_delta > 0 else ''}{k_delta}",
            delta_color="normal" if abs(k_delta) <= crop["ideal_k"] * 0.2 else "inverse")

        st.markdown("---")
        st.markdown('<div class="styled-label"><div>🧪 Fertilizer Recommendations</div></div>', unsafe_allow_html=True)
        recs = []
        if n_delta < -crop["ideal_n"] * 0.15:
            recs.append(FERTILIZER_DB["low_n"])
        elif n_delta > crop["ideal_n"] * 0.2:
            recs.append(FERTILIZER_DB["high_n"])
        if p_delta < -crop["ideal_p"] * 0.15:
            recs.append(FERTILIZER_DB["low_p"])
        elif p_delta > crop["ideal_p"] * 0.2:
            recs.append(FERTILIZER_DB["high_p"])
        if k_delta < -crop["ideal_k"] * 0.15:
            recs.append(FERTILIZER_DB["low_k"])
        elif k_delta > crop["ideal_k"] * 0.2:
            recs.append(FERTILIZER_DB["high_k"])

        if not recs:
            st.success("✅ Your NPK levels are within the ideal range! Maintain current practice.")
        else:
            for rec in recs:
                st.markdown(f'<div class="feature-card" style="text-align:left;margin-bottom:0.8rem;"><h3 style="color:#63ffb6 !important;">{rec["fertilizer"]}</h3><p><b>Dosage:</b> {rec["dosage"]}</p><p>{rec["note"]}</p></div>', unsafe_allow_html=True)

        st.markdown("---")
        opt_ph_lo, opt_ph_hi = crop["optimal_ph"]
        st.markdown('<div class="styled-label"><div>⚗️ pH Adjustment Guidance</div></div>', unsafe_allow_html=True)
        st.markdown(f"**Optimal pH range for {crop['name']}:** {opt_ph_lo} – {opt_ph_hi}")
        st.markdown(f"- If your soil pH is **below {opt_ph_lo}**: {FERTILIZER_DB['low_ph']['fertilizer']} — {FERTILIZER_DB['low_ph']['dosage']}")
        st.markdown(f"- If your soil pH is **above {opt_ph_hi}**: {FERTILIZER_DB['high_ph']['fertilizer']} — {FERTILIZER_DB['high_ph']['dosage']}")

        st.markdown("---")
        st.markdown('<div class="styled-label"><div>🛡️ Pest & Disease Management</div></div>', unsafe_allow_html=True)
        pests = PESTICIDE_DB.get(crop["group"], [])
        if pests:
            pest_df = pd.DataFrame(pests)
            pest_df.columns = ["Pest / Disease", "Recommended Product", "Dosage"]
            pest_df.index = range(1, len(pest_df) + 1)
            st.dataframe(pest_df, use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="styled-label"><div>💧 Irrigation & Planting Tips</div></div>', unsafe_allow_html=True)
        tl, th = crop["optimal_temp"]
        rl, rh = crop["optimal_rain"]
        st.markdown(f"- **Optimal temperature:** {tl}–{th} °C")
        st.markdown(f"- **Optimal rainfall:** {rl}–{rh} mm/season")
        if rh > 200:
            st.markdown("- 💧 **High water crop** — ensure consistent irrigation. Consider drip for efficiency.")
        else:
            st.markdown("- 🌵 **Low-medium water crop** — sprinkler or furrow irrigation recommended.")
        st.markdown(f"- 📅 **Market price (MSP):** ₹{crop['market_price']:,}/quintal")


# =============================================================================
# PAGE 4 — INDIA MAP INFO + ABOUT US
# =============================================================================

def render_india_info():
    st.markdown('<div class="section-header">🗺️ 3D Data Insights — India</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Explore crop yield & soil health intensity across Indian states</div>', unsafe_allow_html=True)

    map_df = pd.DataFrame(REGIONS_DATA)
    map_df["color_r"] = (100 - map_df["soil_health"]).apply(lambda x: int(min(255, x * 3)))
    map_df["color_g"] = map_df["soil_health"].apply(lambda x: int(min(255, x * 2.8)))
    map_df["color_b"] = 80
    column_layer = pdk.Layer("ColumnLayer", data=map_df, get_position=["lon", "lat"],
        get_elevation="yield", elevation_scale=80, radius=35000,
        get_fill_color=["color_r", "color_g", "color_b", 200], pickable=True, auto_highlight=True)
    view = pdk.ViewState(latitude=22.5, longitude=79.5, zoom=4.2, pitch=45, bearing=-15)
    tooltip = {"html": "<div style='font-family:Inter;padding:8px;'><b style='color:#63ffb6;'>{region}</b><br/>🌾 Yield: <b>{yield} kg/ha</b><br/>🩺 Soil Health: <b>{soil_health}%</b><br/>🏆 Top Crop: <b>{top_crop}</b></div>",
        "style": {"backgroundColor": "#1e293b", "color": "#e2e8f0", "border": "1px solid rgba(99,255,182,0.3)", "border-radius": "8px"}}
    st.pydeck_chart(pdk.Deck(layers=[column_layer], initial_view_state=view, tooltip=tooltip,
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"), use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">🏅 Regional Rankings</div>', unsafe_allow_html=True)
    col_y, col_s = st.columns(2)
    with col_y:
        st.markdown("**Top 5 by Crop Yield**")
        top_yield = sorted(REGIONS_DATA, key=lambda x: x["yield"], reverse=True)[:5]
        df_y = pd.DataFrame([{"Region": r["region"], "Yield (kg/ha)": r["yield"], "Top Crop": r["top_crop"]} for r in top_yield])
        df_y.index = range(1, 6)
        st.dataframe(df_y, use_container_width=True)
    with col_s:
        st.markdown("**Top 5 by Soil Health**")
        top_soil = sorted(REGIONS_DATA, key=lambda x: x["soil_health"], reverse=True)[:5]
        df_s = pd.DataFrame([{"Region": r["region"], "Soil Health (%)": r["soil_health"], "Top Crop": r["top_crop"]} for r in top_soil])
        df_s.index = range(1, 6)
        st.dataframe(df_s, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">👥 About Us</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Built with ❤️ for the Pan-India Hackathon 2026</div>', unsafe_allow_html=True)
    ab1, ab2, ab3 = st.columns(3)
    with ab1:
        st.markdown('<div class="feature-card"><div style="font-size:2.2rem;">🎯</div><h3>Our Mission</h3><p>Empower Indian farmers with AI-driven, transparent crop recommendations based on real soil & climate data.</p></div>', unsafe_allow_html=True)
    with ab2:
        st.markdown('<div class="feature-card"><div style="font-size:2.2rem;">🛠️</div><h3>Tech Stack</h3><p>Python • Streamlit • Pydeck • Pandas • ML (Mocked) • Rule-based scoring engine</p></div>', unsafe_allow_html=True)
    with ab3:
        st.markdown('<div class="feature-card"><div style="font-size:2.2rem;">🚀</div><h3>Scalability</h3><p>Modular architecture ready for real ML models, live market APIs, and IoT sensor integration.</p></div>', unsafe_allow_html=True)
    st.markdown("")
    st.markdown('<div style="text-align:center;color:#475569;font-size:0.8rem;padding:2rem 0;">AgriRank AI v2.0 — Pan-India Hackathon 2026 &bull; Demo Build</div>', unsafe_allow_html=True)


# =============================================================================
# SIDEBAR NAVIGATION + PAGE ROUTER
# =============================================================================

if not st.session_state["logged_in"]:
    render_login()
else:
    PAGES = ["Region-wise Top Crops", "Recommendations", "India Map & About"]
    with st.sidebar:
        st.markdown("## 🌾 AgriRank AI")
        st.markdown(f"*Welcome, **{st.session_state['username']}***")
        st.markdown("---")
        st.session_state["current_page"] = st.radio("Navigate", PAGES,
            index=PAGES.index(st.session_state["current_page"]) if st.session_state["current_page"] in PAGES else 0,
            label_visibility="collapsed")
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state["logged_in"] = False
            st.session_state["username"] = ""
            st.rerun()
        st.markdown("---")
        st.caption("Built for Pan-India Hackathon 2026")
        st.caption("v2.0 — Demo Build")

    page = st.session_state["current_page"]
    if page == "Region-wise Top Crops":
        render_region_crops()
    elif page == "Recommendations":
        render_recommendations()
    elif page == "India Map & About":
        render_india_info()
