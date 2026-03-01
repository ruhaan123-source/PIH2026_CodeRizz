# PIH2026_CodeRizz: AgriRank AI

# AI-Assisted Crop Ranking & Soil Health Recommendation System for India

AgriRank AI is a high-performance decision support platform designed to empower Indian farmers with data-driven insights. It leverages advanced Machine Learning (XGBoost) and interactive geospatial visualizations (Pydeck) to provide region-specific crop and soil intelligence.

---

## [Goal] Core Problem
Small and medium-scale Indian farmers often lack access to localized, data-driven advice. This system bridges that gap by:
* Ranking Crops: Determining the most profitable crops for a specific district.
* Yield Prediction: Estimating expected harvest using AI models.
* Soil Diagnostics: Analyzing NPK levels to provide actionable fertilizer recommendations.
* Geospatial Insights: Visualizing agricultural health across an interactive map of India.

---

## [Modules] Module Architecture

### MODULE 1: Data Engineering Layer
Integrated datasets merging historical Indian yield records, district-specific coordinates, and ideal crop requirements.
* Optimized Storage: district_coords.csv shrunk from 44MB to 34KB for lightning-fast web performance.

### MODULE 2: Yield Prediction Model (XGBoost)
A high-accuracy regression model trained on soil pH, Temperature, Rainfall, and NPK levels.
* Form Factor: Converted to Universal Binary JSON (.ubj) for 35% smaller file size and faster inference.

### MODULE 3: Rule-Based Compatibility Engine
Computes a suitability score by comparing actual farm parameters against ideal crop conditions.
* Formula: Score = 1 - |actual - ideal| / tolerance

### MODULE 4: Ranking Engine
Synthesizes predicted yield and compatibility into a final actionable rank.
* Logic: Final Score = (Alpha) * Normalized Yield + (Beta) * Compatibility

### MODULE 5: Soil Health Diagnostic System
Generates precise fertilizer dosages (e.g., Urea, DAP, MOP) and soil amendments based on real-time NPK and environmental inputs.

### MODULE 6: Visualization Interface
A sleek, modern dashboard built with Streamlit and Pydeck, featuring a 3D regional map and responsive analytics.

---

## [Project] Project Structure
```text
.
├── app_combined.py       # Main Platform Entry
├── crop_inference.py     # AI Ranking Logic
├── predict_fertilizer.py # Soil Analysis Logic
├── requirements.txt      # Dependency List
└── assets/  
    ├── css/              # Premium Styling
    ├── data/             # Shrunk & Optimized CSVs
    └── models/           # XGBoost Binary & Encoders
```

---

## [Deploy] Deployment Guide

### Streamlit Community Cloud (Recommended)
1. GitHub: Push this project to a GitHub repository.
2. Connect: Log in to share.streamlit.io.
3. Deploy: Select app_combined.py as the main file path.

### Local Execution
```powershell
pip install -r requirements.txt
streamlit run app_combined.py
```

---

## [Access] Access
The system currently features a simplified Developer Login.
* Username: admin
* Password: (Any or blank)
