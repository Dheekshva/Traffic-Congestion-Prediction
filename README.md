# Traffic Congestion Prediction — Bengaluru

A deep learning system for predicting urban traffic congestion levels (Low / Medium / High) using a CNN-LSTM architecture, with explainability and class balancing.

---

## Overview

Existing traffic prediction models focus on accuracy but rarely address class imbalance, external factors like weather, or explainability. This project tackles all three by combining CNN-LSTM with SMOTE-based class balancing and permutation-based feature importance.

---

## Key Features

- CNN-LSTM model for spatio-temporal traffic pattern learning
- SMOTE to handle class imbalance across congestion levels
- Permutation-based feature importance for explainability
- Bengaluru-specific dataset augmented using TomTom Traffic Index 2022/2023
- Live data collection pipeline using TomTom Traffic API across 5 Bengaluru junctions
- Streamlit web application for real-time congestion prediction

---

## Models

| Version | Dataset | Test Accuracy |
|---|---|---|
| Version 1 | Metro Interstate Traffic Volume (Kaggle) | 91.89% |
| Version 2 | Bengaluru Augmented Dataset | 89.13% |

---

## Baseline Comparison

| Model | Accuracy |
|---|---|
| Random Forest (baseline) | 91.38% |
| LSTM Only (baseline) | 91.64% |
| CNN-LSTM (proposed) | 91.89% |

---

## Gaps Addressed

| Gap in Literature | Our Solution |
|---|---|
| Class imbalance ignored | SMOTE oversampling |
| No weather or event features | 10 features including temp, rain, holiday |
| No explainability | Permutation-based feature importance |
| No Bengaluru-specific model | Domain-augmented dataset |

---

## Tech Stack

- Python, TensorFlow, Keras
- scikit-learn, imbalanced-learn
- Streamlit
- TomTom Traffic API, Open-Meteo API

---

## Live Data Pipeline

Real-time traffic and weather data collected every 15 minutes from:

- Silk Board Junction
- MG Road
- Hebbal Flyover
- Electronic City Toll
- Marathahalli Bridge

---

## Project Structure

    traffic-congestion-prediction/
    ├── notebook/
    │   └── traffic_cong_pred_clean.ipynb
    ├── app/
    │   ├── version1_metro/
    │   │   └── app.py
    │   └── version2_bengaluru/
    │       └── app.py
    └── data/
        └── Bengaluru_Augmented_Traffic.csv

---

## Data Sources

- Metro Interstate Traffic Volume Dataset — Kaggle
- TomTom Traffic Index 2022 — Bengaluru ranked 2nd globally
- TomTom Traffic Index 2023 — 63% congestion level, 18 km/hr average speed
- Uber Movement Study — Weekend travel 57% shorter than weekday mornings

---

## References

1. TomTom Traffic Index 2022 — https://www.tomtom.com/traffic-index/
2. TomTom Traffic Index 2023 — https://www.tomtom.com/traffic-index/
3. Uber Movement Study, Bengaluru — Whitefield Metro Impact on Peak Hour Travel
4. Lv et al. (2015) — Traffic Flow Prediction with Big Data: A Deep Learning Approach, IEEE TITS
5. Ma et al. (2015) — Long Short-Term Memory Neural Network for Traffic Speed Prediction, Elsevier TRC
6. Bogaerts et al. (2020) — A Graph CNN-LSTM Neural Network for Traffic Forecasting, Elsevier TRC
