# PIH2026_CodeRizz

#  AI-Assisted Crop Ranking & Soil Health Recommendation System for India

##  Problem Title
**AI-Assisted Crop Ranking & Soil Health Recommendation System for India**

---

##  Core Problem
Small and medium-scale Indian farmers lack a data-driven, region-specific decision support system that:
* Ranks crops by suitability
* Predicts expected yield
* Diagnoses soil health
* Provides actionable recommendations

### Core Requirements
1. Predict yield using ML.
2. Compute compatibility using rule logic.
3. Rank crops.
4. Provide soil health recommendations.
5. Visualize results on an interactive India map.

---

##  MODULE-WISE DISTRIBUTABLE PROBLEMS

###  MODULE 1: Data Engineering Layer
**Problem Statement**: Design a structured dataset pipeline that merges:
* Crop requirements
* Soil health data (NPK, pH)
* Climate averages (temperature, rainfall)
* Historical crop yield data (India-specific)

**Deliverables**: Clean CSV files, Unified dataset schema, Feature-ready training dataset.

###  MODULE 2: Yield Prediction Model (ML)
**Problem Statement**: Build a regression model using XGBoost to predict crop yield based on NPK, Soil pH, Temperature, Rainfall, and Humidity.

**Deliverables**: Trained XGBoost model, Model evaluation (RMSE, R²), Feature importance chart.

### 🟣 MODULE 3: Rule-Based Compatibility Engine
**Problem Statement**: Develop a compatibility scoring system using the formula:
`Score = 1 - |actual - ideal| / tolerance`

**Deliverables**: Compatibility score (0–1), Factor-wise breakdown.

###  MODULE 4: Ranking Engine
**Problem Statement**: Combine predicted yield and compatibility score into a final ranking score.
`Final Score = α * Normalized Predicted Yield + β * Compatibility Score`

**Deliverables**: Ranked list of crops, Top 5 recommendations, Score breakdown.

###  MODULE 5: Soil Health Diagnostic System
**Problem Statement**: Generate actionable soil health advice based on detected deficiencies (e.g., Nitrogen recommendations, pH adjustments).

**Deliverables**: Crop-specific recommendations, Health warning indicators.

###  MODULE 6: Visualization & Interface
**Problem Statement**: Build a user interface with Streamlit and Plotly Choropleth that accepts region input and displays results on an interactive India map.

**Deliverables**: Interactive dashboard, Map visualization, Feature importance display.

---

##  SYSTEM FLOW
1. **Region Selected**
2. **Load Soil + Climate Data**
3. **For Each Crop**:
   - Predict Yield (Module 2)
   - Compute Compatibility (Module 3)
4. **Ranking Engine** (Module 4)
5. **Soil Health Advice** (Module 5)
6. **Dashboard + Map** (Module 6)

---
