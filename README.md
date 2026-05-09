# Traffic Congestion Prediction — Bengaluru

A deep learning system for predicting urban traffic congestion levels 
(Low / Medium / High) using a CNN-LSTM architecture, with explainability 
and class balancing.

## Overview

Existing traffic prediction models focus on accuracy but rarely address 
class imbalance, external factors like weather, or explainability. This 
project tackles all three.

## Key Features

- CNN-LSTM model for spatio-temporal traffic pattern learning
- SMOTE to handle class imbalance across congestion levels
- Permutation-based feature importance for explainability
- Bengaluru-specific dataset augmented using TomTom Traffic Index 2022/2023
- Live data collection pipeline using TomTom Traffic API (5 Bengaluru junctions)
- Streamlit web application for real-time congestion prediction

## Models

| Version | Dataset | Accuracy |
|---|---|---|
| Version 1 | Metro Interstate Traffic Volume | 91.89% |
| Version 2 | Bengaluru Augmented | 89.13% |

## Results

| Model | Accuracy |
|---|---|
| Random Forest (baseline) | 91.38% |
| LSTM Only (baseline) | 91.64% |
| CNN-LSTM (proposed) | 91.89% |

## Tech Stack

- Python, TensorFlow, Keras
- scikit-learn, imbalanced-learn
- Streamlit
- TomTom Traffic API, Open-Meteo API

## Data Sources

- Metro Interstate Traffic Volume Dataset (Kaggle)
- TomTom Traffic Index 2022/2023
- Uber Movement Study — Bengaluru

## Live Data Pipeline

Real-time traffic and weather data collected every 15 minutes from:
- Silk Board Junction
- MG Road
- Hebbal Flyover
- Electronic City Toll
- Marathahalli Bridge

## Project Structure
