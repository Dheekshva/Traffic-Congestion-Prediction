import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization

# ── Constants ─────────────────────────────────────────────────────
SEQUENCE_LENGTH = 24
N_FEATURES      = 10
NUM_CLASSES     = 3

FEATURES = [
    'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday',
    'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_encoded'
]

WEATHER_OPTIONS = {
    'Clear': 0, 'Clouds': 1, 'Drizzle': 2, 'Fog': 3,
    'Haze': 4, 'Mist': 5, 'Rain': 6, 'Snow': 7,
    'Squall': 8, 'Thunderstorm': 9
}

LOCATIONS = [
    'Silk Board Junction',
    'MG Road',
    'Hebbal Flyover',
    'Electronic City Toll',
    'Marathahalli Bridge',
    'Outer Ring Road',
    'Whitefield',
    'Koramangala'
]

MONSOON_MONTHS = [6, 7, 8, 9]

# ── Model loader ──────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = Sequential([
    Conv1D(128, kernel_size=3, activation='relu',
           input_shape=(SEQUENCE_LENGTH, N_FEATURES)),
    BatchNormalization(),
    Conv1D(128, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    LSTM(256, return_sequences=True),
    Dropout(0.2),
    LSTM(128, return_sequences=False),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.load_weights('blr_weights.weights.h5')

    with open('scaler_blr.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('le_blr.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, scaler, le

model, scaler, le = load_model()

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Bengaluru Traffic Predictor",
    page_icon="",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main {
        background-color: #f7f6f3;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    h1, h2, h3 {
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        letter-spacing: -0.02em;
    }

    .header-section {
        border-bottom: 1px solid #e0ddd6;
        padding-bottom: 1.5rem;
        margin-bottom: 2rem;
    }

    .header-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1a1a1a;
        margin: 0;
        letter-spacing: -0.03em;
    }

    .header-sub {
        font-size: 0.85rem;
        color: #888;
        margin-top: 0.3rem;
        font-family: 'DM Mono', monospace;
    }

    .result-high {
        background: #fff0f0;
        border-left: 3px solid #c0392b;
        padding: 1.2rem 1.5rem;
        border-radius: 4px;
        margin: 1rem 0;
    }

    .result-medium {
        background: #fffbf0;
        border-left: 3px solid #d4820a;
        padding: 1.2rem 1.5rem;
        border-radius: 4px;
        margin: 1rem 0;
    }

    .result-low {
        background: #f0fff4;
        border-left: 3px solid #27ae60;
        padding: 1.2rem 1.5rem;
        border-radius: 4px;
        margin: 1rem 0;
    }

    .result-label {
        font-size: 1.4rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        margin: 0;
    }

    .result-confidence {
        font-size: 0.8rem;
        font-family: 'DM Mono', monospace;
        color: #666;
        margin-top: 0.3rem;
    }

    .info-card {
        background: white;
        border: 1px solid #e8e5df;
        border-radius: 6px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
    }

    .info-card-title {
        font-size: 0.7rem;
        font-family: 'DM Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #888;
        margin-bottom: 0.4rem;
    }

    .info-card-value {
        font-size: 1rem;
        font-weight: 500;
        color: #1a1a1a;
    }

    .feature-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid #f0ede8;
    }

    .feature-row:last-child {
        border-bottom: none;
    }

    .feature-name {
        font-size: 0.85rem;
        color: #444;
        font-family: 'DM Mono', monospace;
    }

    .feature-bar-container {
        width: 140px;
        height: 4px;
        background: #e8e5df;
        border-radius: 2px;
        overflow: hidden;
    }

    .feature-bar-fill {
        height: 100%;
        background: #1a1a1a;
        border-radius: 2px;
    }

    .feature-score {
        font-size: 0.75rem;
        font-family: 'DM Mono', monospace;
        color: #888;
        width: 50px;
        text-align: right;
    }

    .warning-banner {
        background: #fff8e6;
        border: 1px solid #f0c040;
        border-radius: 4px;
        padding: 0.8rem 1.2rem;
        font-size: 0.85rem;
        color: #7a5c00;
        margin-bottom: 1rem;
    }

    .section-label {
        font-size: 0.7rem;
        font-family: 'DM Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #aaa;
        margin-bottom: 1rem;
        margin-top: 2rem;
    }

    .prob-row {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.4rem 0;
    }

    .prob-label {
        font-size: 0.8rem;
        font-family: 'DM Mono', monospace;
        width: 70px;
        color: #555;
    }

    .prob-bar-wrap {
        flex: 1;
        height: 6px;
        background: #ece9e3;
        border-radius: 3px;
        overflow: hidden;
    }

    .prob-pct {
        font-size: 0.75rem;
        font-family: 'DM Mono', monospace;
        color: #888;
        width: 40px;
        text-align: right;
    }

    .stButton button {
        background: #1a1a1a;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.6rem 2rem;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.9rem;
        font-weight: 500;
        cursor: pointer;
        width: 100%;
        transition: background 0.2s;
    }

    .stButton button:hover {
        background: #333;
    }

    .stSelectbox label, .stSlider label, .stToggle label {
        font-size: 0.8rem !important;
        font-family: 'DM Mono', monospace !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: #888 !important;
    }

    footer { display: none; }
    #MainMenu { display: none; }
    header { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div class="header-section">
    <p class="header-title">Bengaluru Traffic Congestion Predictor</p>
    <p class="header-sub">CNN-LSTM model — trained on Bengaluru-augmented data &nbsp;·&nbsp; accuracy 89.13%</p>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.markdown('<p class="section-label">Conditions</p>', unsafe_allow_html=True)

    location    = st.selectbox("Location", LOCATIONS)
    day_of_week = st.selectbox("Day", ['Monday','Tuesday','Wednesday',
                                        'Thursday','Friday','Saturday','Sunday'])
    hour        = st.slider("Hour of day", 0, 23, 8)
    month       = st.slider("Month", 1, 12, 6)
    temp        = st.slider("Temperature (°C)", 10, 45, 28)
    weather     = st.selectbox("Weather", list(WEATHER_OPTIONS.keys()))
    rain        = st.slider("Rainfall (mm)", 0.0, 50.0, 0.0)
    clouds      = st.slider("Cloud cover (%)", 0, 100, 40)
    is_holiday  = st.toggle("Public holiday", value=False)

    day_num    = ['Monday','Tuesday','Wednesday',
                  'Thursday','Friday','Saturday','Sunday'].index(day_of_week)
    is_weekend = 1 if day_of_week in ['Saturday','Sunday'] else 0
    snow       = 0.0  # Bengaluru doesn't snow

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Predict", use_container_width=True)

with col_result:
    st.markdown('<p class="section-label">Prediction</p>', unsafe_allow_html=True)

    if predict_btn:

        # Monsoon warning
        if month in MONSOON_MONTHS and rain > 0:
            st.markdown("""
            <div class="warning-banner">
                Monsoon conditions detected. Bengaluru roads typically experience
                significantly higher congestion during heavy rainfall — especially
                on Outer Ring Road and Silk Board Junction.
            </div>
            """, unsafe_allow_html=True)

        # Friday evening warning
        if day_of_week == 'Friday' and 18 <= hour <= 19:
            st.markdown("""
            <div class="warning-banner">
                Friday 6–7 pm is historically the worst congestion window
                in Bengaluru (TomTom Traffic Index 2022).
            </div>
            """, unsafe_allow_html=True)

        # Build sequence
        input_row = np.array([[
            hour, day_num, month, is_weekend, int(is_holiday),
            temp, rain, snow, clouds, WEATHER_OPTIONS[weather]
        ]])
        input_scaled = scaler.transform(input_row)

        sequence = []
        for i in range(SEQUENCE_LENGTH):
            hour_i    = (hour - (SEQUENCE_LENGTH - 1 - i)) % 24
            row       = input_scaled.copy()
            row[0][0] = hour_i / 23.0
            sequence.append(row[0])

        input_seq  = np.array(sequence).reshape(1, SEQUENCE_LENGTH, N_FEATURES)
        pred_prob  = model.predict(input_seq, verbose=0)[0]
        pred_class = np.argmax(pred_prob)
        pred_label = le.classes_[pred_class]
        confidence = pred_prob[pred_class] * 100

        # Result card
        css_class = {
            'High':   'result-high',
            'Medium': 'result-medium',
            'Low':    'result-low'
        }[pred_label]

        label_text = {
            'High':   'High Congestion',
            'Medium': 'Moderate Congestion',
            'Low':    'Low Congestion'
        }[pred_label]

        st.markdown(f"""
        <div class="{css_class}">
            <p class="result-label">{label_text}</p>
            <p class="result-confidence">{location} &nbsp;·&nbsp; {confidence:.1f}% confidence</p>
        </div>
        """, unsafe_allow_html=True)

        # Probability breakdown
        st.markdown('<p class="section-label" style="margin-top:1.5rem">Probability breakdown</p>',
                    unsafe_allow_html=True)

        bar_colors = {'High': '#c0392b', 'Medium': '#d4820a', 'Low': '#27ae60'}
        for i, cls in enumerate(le.classes_):
            pct = pred_prob[i] * 100
            color = bar_colors.get(cls, '#333')
            st.markdown(f"""
            <div class="prob-row">
                <span class="prob-label">{cls}</span>
                <div class="prob-bar-wrap">
                    <div style="width:{pct}%; height:100%; background:{color}; border-radius:3px;"></div>
                </div>
                <span class="prob-pct">{pct:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

        # What to expect
        st.markdown('<p class="section-label" style="margin-top:1.5rem">What to expect</p>',
                    unsafe_allow_html=True)

        advice = {
            'High': [
                ('Estimated delay', '+25–45 min for 10 km'),
                ('Avg speed', 'Below 18 km/hr'),
                ('Suggestion', 'Delay travel or use Metro'),
            ],
            'Medium': [
                ('Estimated delay', '+10–20 min for 10 km'),
                ('Avg speed', '25–35 km/hr'),
                ('Suggestion', 'Allow extra travel time'),
            ],
            'Low': [
                ('Estimated delay', 'Minimal'),
                ('Avg speed', 'Above 40 km/hr'),
                ('Suggestion', 'Good time to travel'),
            ]
        }[pred_label]

        for title, value in advice:
            st.markdown(f"""
            <div class="info-card">
                <p class="info-card-title">{title}</p>
                <p class="info-card-value">{value}</p>
            </div>
            """, unsafe_allow_html=True)

        # Feature importance
        st.markdown('<p class="section-label" style="margin-top:1.5rem">Feature influence</p>',
                    unsafe_allow_html=True)

        importance = [
            ('hour',            0.5365),
            ('is_weekend',      0.0967),
            ('day_of_week',     0.0682),
            ('weather',         0.0190),
            ('is_holiday',      0.0079),
            ('temperature',     0.0068),
            ('cloud_cover',     0.0067),
            ('month',           0.0022),
            ('rainfall',        0.0003),
        ]

        max_val = importance[0][1]
        st.markdown('<div class="info-card" style="padding: 0.8rem 1.5rem">',
                    unsafe_allow_html=True)
        for feat, score in importance:
            bar_width = int((score / max_val) * 100)
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                        padding:0.5rem 0; border-bottom:1px solid #f0ede8;">
                <span style="font-size:0.85rem; color:#444; font-family:'DM Mono',monospace;
                             width:120px">{feat}</span>
                <div style="width:140px; height:4px; background:#e8e5df; border-radius:2px; overflow:hidden">
                    <div style="width:{bar_width}%; height:100%; background:#1a1a1a; border-radius:2px"></div>
                </div>
                <span style="font-size:0.75rem; font-family:'DM Mono',monospace; color:#888;
                             width:50px; text-align:right">{score:.4f}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <p style="font-size:0.72rem; color:#aaa; font-family:'DM Mono',monospace; margin-top:0.5rem">
        Permutation-based feature importance · TomTom Traffic Index 2022/2023
        </p>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="color:#bbb; font-size:0.9rem; margin-top:2rem; font-family:'DM Mono',monospace">
            Set conditions and click Predict.
        </div>
        """, unsafe_allow_html=True)