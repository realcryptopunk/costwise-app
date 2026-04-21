"""
CostWise — Regional Construction Cost Estimator
A practical web application powered by the CatBoost regional-aware model.
"""

from flask import Flask, render_template, jsonify, request
from catboost import CatBoostRegressor
import numpy as np
import json
from pathlib import Path

app = Flask(__name__)

# ── Load model and config ─────────────────────────────────────────────────────

MODEL_PATH = Path(__file__).parent / "model" / "model_b_regional.cbm"
model = None

# City data with CCI values (from our training data)
CITIES = {
    # Northeast
    "New York, NY": {"mat": 118.2, "labor": 148.5, "equip": 105.3, "region": "Northeast", "state": "NY"},
    "Boston, MA": {"mat": 112.8, "labor": 138.7, "equip": 103.8, "region": "Northeast", "state": "MA"},
    "Philadelphia, PA": {"mat": 110.5, "labor": 132.4, "equip": 102.1, "region": "Northeast", "state": "PA"},
    "Newark, NJ": {"mat": 113.1, "labor": 141.2, "equip": 104.5, "region": "Northeast", "state": "NJ"},
    "Hartford, CT": {"mat": 109.4, "labor": 125.8, "equip": 102.7, "region": "Northeast", "state": "CT"},
    "Pittsburgh, PA": {"mat": 104.2, "labor": 114.6, "equip": 100.8, "region": "Northeast", "state": "PA"},
    "Washington, DC": {"mat": 107.3, "labor": 119.5, "equip": 102.4, "region": "Northeast", "state": "DC"},
    "Baltimore, MD": {"mat": 104.8, "labor": 112.3, "equip": 101.1, "region": "Northeast", "state": "MD"},
    # Southeast
    "Atlanta, GA": {"mat": 100.5, "labor": 89.4, "equip": 99.2, "region": "Southeast", "state": "GA"},
    "Miami, FL": {"mat": 105.3, "labor": 95.7, "equip": 100.8, "region": "Southeast", "state": "FL"},
    "Charlotte, NC": {"mat": 99.1, "labor": 82.6, "equip": 98.7, "region": "Southeast", "state": "NC"},
    "Nashville, TN": {"mat": 99.8, "labor": 85.3, "equip": 99.1, "region": "Southeast", "state": "TN"},
    "Orlando, FL": {"mat": 101.2, "labor": 88.1, "equip": 99.5, "region": "Southeast", "state": "FL"},
    "Tampa, FL": {"mat": 100.8, "labor": 86.9, "equip": 99.3, "region": "Southeast", "state": "FL"},
    "Charleston, SC": {"mat": 99.3, "labor": 81.8, "equip": 98.6, "region": "Southeast", "state": "SC"},
    "Richmond, VA": {"mat": 100.1, "labor": 87.5, "equip": 99.0, "region": "Southeast", "state": "VA"},
    # Midwest
    "Chicago, IL": {"mat": 107.8, "labor": 128.5, "equip": 103.2, "region": "Midwest", "state": "IL"},
    "Detroit, MI": {"mat": 104.6, "labor": 118.2, "equip": 101.5, "region": "Midwest", "state": "MI"},
    "Minneapolis, MN": {"mat": 106.3, "labor": 119.7, "equip": 102.1, "region": "Midwest", "state": "MN"},
    "Columbus, OH": {"mat": 100.2, "labor": 97.5, "equip": 99.8, "region": "Midwest", "state": "OH"},
    "Indianapolis, IN": {"mat": 99.5, "labor": 95.8, "equip": 99.4, "region": "Midwest", "state": "IN"},
    "Kansas City, MO": {"mat": 101.3, "labor": 102.4, "equip": 100.2, "region": "Midwest", "state": "MO"},
    "Milwaukee, WI": {"mat": 104.1, "labor": 115.3, "equip": 101.3, "region": "Midwest", "state": "WI"},
    "St. Louis, MO": {"mat": 101.8, "labor": 108.4, "equip": 100.6, "region": "Midwest", "state": "MO"},
    # Southwest
    "Dallas, TX": {"mat": 98.4, "labor": 80.1, "equip": 99.2, "region": "Southwest", "state": "TX"},
    "Houston, TX": {"mat": 99.7, "labor": 83.5, "equip": 99.8, "region": "Southwest", "state": "TX"},
    "Phoenix, AZ": {"mat": 100.3, "labor": 86.7, "equip": 99.5, "region": "Southwest", "state": "AZ"},
    "Austin, TX": {"mat": 98.9, "labor": 81.3, "equip": 99.3, "region": "Southwest", "state": "TX"},
    "Denver, CO": {"mat": 103.5, "labor": 98.4, "equip": 100.8, "region": "Southwest", "state": "CO"},
    "Las Vegas, NV": {"mat": 104.2, "labor": 105.6, "equip": 101.0, "region": "Southwest", "state": "NV"},
    "Oklahoma City, OK": {"mat": 97.2, "labor": 78.3, "equip": 98.5, "region": "Southwest", "state": "OK"},
    # West
    "San Francisco, CA": {"mat": 117.5, "labor": 155.2, "equip": 106.1, "region": "West", "state": "CA"},
    "Los Angeles, CA": {"mat": 112.3, "labor": 135.8, "equip": 104.2, "region": "West", "state": "CA"},
    "Seattle, WA": {"mat": 110.8, "labor": 125.4, "equip": 103.5, "region": "West", "state": "WA"},
    "Portland, OR": {"mat": 107.6, "labor": 116.8, "equip": 102.3, "region": "West", "state": "OR"},
    "San Diego, CA": {"mat": 109.4, "labor": 128.7, "equip": 103.8, "region": "West", "state": "CA"},
    "Sacramento, CA": {"mat": 108.2, "labor": 126.3, "equip": 103.1, "region": "West", "state": "CA"},
    "Honolulu, HI": {"mat": 121.5, "labor": 138.4, "equip": 108.2, "region": "West", "state": "HI"},
    "San Jose, CA": {"mat": 115.8, "labor": 150.3, "equip": 105.8, "region": "West", "state": "CA"},
}

PROJECT_TYPES = ["Commercial", "Residential", "Industrial", "Institutional", "Infrastructure"]

# Current macro values (2025 averages from FRED)
MACRO_DEFAULTS = {
    "ppi_construction_materials": 280.0,
    "ppi_cement": 350.0,
    "ppi_steel_mill": 290.0,
    "ppi_lumber": 260.0,
    "real_gdp": 22500.0,
    "cpi_all_urban": 315.0,
    "unemployment_rate": 4.1,
    "mortgage_30yr": 6.8,
    "building_permits": 1450.0,
    "housing_starts": 1380.0,
    "regional_cpi": 310.0,
}


def load_model():
    global model
    if MODEL_PATH.exists():
        model = CatBoostRegressor()
        model.load_model(str(MODEL_PATH))
        print(f"✅ Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠️ Model not found at {MODEL_PATH} — predictions will use fallback formula")


def predict_cost(city, project_type, area_sqft, year=2026):
    """Predict cost per sqft using Model B or fallback."""
    
    city_data = CITIES[city]
    mat_cci = city_data["mat"]
    labor_cci = city_data["labor"]
    equip_cci = city_data["equip"]
    weighted_cci = 0.45 * mat_cci + 0.40 * labor_cci + 0.15 * equip_cci
    
    # Derived features
    formwork_rate = 5.5 * (mat_cci / 100)
    concrete_rate = 7.5 * (mat_cci / 100)
    cci_labor_premium = labor_cci - mat_cci
    cci_deviation = weighted_cci - 100
    combined_material_rate = formwork_rate + concrete_rate
    log_area = np.log1p(area_sqft)
    year_num = year - 2015
    ppi_yoy_change = 0.03
    
    if model is not None:
        # Build feature vector matching Model B training order
        features = {
            "area_sqft": area_sqft,
            "formwork_rate": formwork_rate,
            "concrete_rate": concrete_rate,
            "project_type": project_type,
            "region": city_data["region"],
            "state": city_data["state"],
            "year": year,
            "mat_cci": mat_cci,
            "labor_cci": labor_cci,
            "equip_cci": equip_cci,
            "weighted_cci": weighted_cci,
            "cci_labor_premium": cci_labor_premium,
            "cci_deviation": cci_deviation,
            "combined_material_rate": combined_material_rate,
            "log_area": log_area,
            "year_num": year_num,
            **MACRO_DEFAULTS,
            "ppi_yoy_change": ppi_yoy_change,
        }
        
        import pandas as pd
        df = pd.DataFrame([features])
        for col in ["project_type", "region", "state"]:
            df[col] = df[col].astype(str)
        
        # Get feature order from model
        model_features = model.feature_names_
        # Reorder and fill missing
        for f in model_features:
            if f not in df.columns:
                df[f] = 0
        df = df[model_features]
        
        cost_per_sqft = float(model.predict(df)[0])
    else:
        # Fallback formula
        type_mult = {"Commercial": 1.0, "Residential": 0.85, "Industrial": 0.75,
                     "Institutional": 1.15, "Infrastructure": 1.30}[project_type]
        scale_factor = (area_sqft / 50000) ** (-0.08)
        cost_per_sqft = (weighted_cci / 100) * 185 * type_mult * scale_factor
    
    total_cost = cost_per_sqft * area_sqft
    
    # Comparison: what would national avg predict?
    national_cci = 100
    national_weighted = national_cci
    if model is not None:
        # Quick national estimate using avg CCI
        nat_features = features.copy()
        nat_features["mat_cci"] = 100
        nat_features["labor_cci"] = 100
        nat_features["equip_cci"] = 100
        nat_features["weighted_cci"] = 100
        nat_features["cci_labor_premium"] = 0
        nat_features["cci_deviation"] = 0
        nat_df = pd.DataFrame([nat_features])
        for col in ["project_type", "region", "state"]:
            nat_df[col] = nat_df[col].astype(str)
        for f in model_features:
            if f not in nat_df.columns:
                nat_df[f] = 0
        nat_df = nat_df[model_features]
        national_cost = float(model.predict(nat_df)[0])
    else:
        type_mult = {"Commercial": 1.0, "Residential": 0.85, "Industrial": 0.75,
                     "Institutional": 1.15, "Infrastructure": 1.30}[project_type]
        scale_factor = (area_sqft / 50000) ** (-0.08)
        national_cost = 185 * type_mult * scale_factor
    
    regional_premium = ((cost_per_sqft - national_cost) / national_cost) * 100
    
    return {
        "cost_per_sqft": round(cost_per_sqft, 2),
        "total_cost": round(total_cost, 0),
        "national_avg_cost": round(national_cost, 2),
        "regional_premium_pct": round(regional_premium, 1),
        "city": city,
        "region": city_data["region"],
        "state": city_data["state"],
        "project_type": project_type,
        "area_sqft": area_sqft,
        "mat_cci": mat_cci,
        "labor_cci": labor_cci,
        "equip_cci": equip_cci,
        "weighted_cci": round(weighted_cci, 1),
        "model_used": "CatBoost Regional-Aware (Model B)" if model else "Fallback Formula",
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/cities")
def get_cities():
    cities_by_region = {}
    for city, data in sorted(CITIES.items()):
        region = data["region"]
        if region not in cities_by_region:
            cities_by_region[region] = []
        cities_by_region[region].append({
            "name": city,
            "weighted_cci": round(0.45*data["mat"] + 0.40*data["labor"] + 0.15*data["equip"], 1)
        })
    return jsonify(cities_by_region)

@app.route("/api/estimate", methods=["POST"])
def estimate():
    data = request.json
    city = data.get("city")
    project_type = data.get("project_type", "Commercial")
    area_sqft = float(data.get("area_sqft", 50000))
    
    if city not in CITIES:
        return jsonify({"error": f"City not found: {city}"}), 400
    if project_type not in PROJECT_TYPES:
        return jsonify({"error": f"Invalid project type: {project_type}"}), 400
    
    result = predict_cost(city, project_type, area_sqft)
    return jsonify(result)

@app.route("/api/compare", methods=["POST"])
def compare():
    """Compare costs across multiple cities for the same project."""
    data = request.json
    cities = data.get("cities", [])
    project_type = data.get("project_type", "Commercial")
    area_sqft = float(data.get("area_sqft", 50000))
    
    results = []
    for city in cities:
        if city in CITIES:
            results.append(predict_cost(city, project_type, area_sqft))
    
    results.sort(key=lambda x: x["cost_per_sqft"])
    return jsonify(results)


if __name__ == "__main__":
    load_model()
    print("\n🏗️ CostWise — Regional Construction Cost Estimator")
    print("   http://localhost:5001\n")
    app.run(debug=True, port=5001)
