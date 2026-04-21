# CostWise — Regional Construction Cost Estimator

A web application that predicts construction costs using a regional-aware CatBoost ML model trained on city-level cost indices and macroeconomic data.

![CostWise](https://img.shields.io/badge/ML-CatBoost_R²_0.920-00d2ff) ![Cities](https://img.shields.io/badge/Coverage-50_US_Cities-3a7bd5) ![Regions](https://img.shields.io/badge/Regions-5_Census-00c853)

## Features

- **Instant cost estimates** for 50 US cities across 5 regions
- **Regional CCI breakdown** — material, labor, equipment indices
- **National comparison** — see how your city compares to the national average
- **Multi-city comparison** — rank all cities side by side
- **ML-powered** — CatBoost model with R² = 0.920 (regional-aware, 28 features)

## Quick Start

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5001
```

## How It Works

1. Select a city, project type, and area
2. The app feeds your inputs + the city's CCI indices + current macro data into the trained CatBoost model
3. Returns: predicted cost/sqft, total cost, regional premium vs national average, CCI breakdown

## Model

The underlying model is from [cost-estimation-ml](https://github.com/realcryptopunk/cost-estimation-ml):
- **Model B (Regional-Aware)**: 28 features including CCI sub-indices, FRED PPI/CPI/GDP, derived features
- **R² = 0.920**, RMSE = $18.03/sqft, MAPE = 6.30%
- Trained on 2,984 projects across 50 US cities

## Tech Stack

- **Backend**: Flask + CatBoost
- **Frontend**: Vanilla HTML/CSS/JS (no frameworks)
- **Model**: CatBoost `.cbm` file
