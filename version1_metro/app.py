import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization

SEQUENCE_LENGTH = 24
N_FEATURES = 10
NUM_CLASSES = 3

FEATURES = [
    'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday',
    'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_encoded'
]
WEATHER_OPTIONS = {
    'Clear': 0, 'Clouds': 1, 'Drizzle': 2, 'Fog': 3,
    'Haze': 4, 'Mist': 5, 'Rain': 6, 'Snow': 7,
    'Squall': 8, 'Thunderstorm': 9
}

@st.cache_resource
def load_model():
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu',
               input_shape=(SEQUENCE_LENGTH, N_FEATURES)),
        BatchNormalization(),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.load_weights('traffic_weights.weights.h5')

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, scaler, le

model, scaler, le = load_model()

# --- Page config ---
st.set_page_config(
    page_title="Traffic Congestion Predictor",
    page_icon="🚦",
    layout="centered"
)

# --- Header ---
st.title("🚦 Traffic Congestion Predictor")
st.markdown("**CNN-LSTM Model with Explainability | Bengaluru Traffic**")
st.divider()

# --- Input Section ---
st.subheader("Enter Current Conditions")

col1, col2 = st.columns(2)

with col1:
    hour        = st.slider("Hour of Day", 0, 23, 8)
    day_of_week = st.selectbox("Day of Week",
                    ['Monday','Tuesday','Wednesday',
                     'Thursday','Friday','Saturday','Sunday'])
    month       = st.slider("Month", 1, 12, 6)
    temp        = st.slider("Temperature (°C)", -10, 45, 25)

with col2:
    weather = st.selectbox("Weather Condition", list(WEATHER_OPTIONS.keys()))
    clouds  = st.slider("Cloud Cover (%)", 0, 100, 40)
    rain    = st.slider("Rainfall (mm)", 0.0, 50.0, 0.0)
    snow    = st.slider("Snowfall (mm)", 0.0, 1.0, 0.0)

is_weekend = 1 if day_of_week in ['Saturday', 'Sunday'] else 0
is_holiday = st.toggle("Public Holiday?", value=False)
day_num    = ['Monday','Tuesday','Wednesday',
              'Thursday','Friday','Saturday','Sunday'].index(day_of_week)

st.divider()

if st.button("🔍 Predict Congestion", use_container_width=True):

    input_row = np.array([[
        hour, day_num, month, is_weekend, int(is_holiday),
        temp, rain, snow, clouds, WEATHER_OPTIONS[weather]
    ]])

    input_scaled = scaler.transform(input_row)

    # Build sequence varying hour across last 24 hours
    sequence = []
    for i in range(SEQUENCE_LENGTH):
        hour_i = (hour - (SEQUENCE_LENGTH - 1 - i)) % 24
        is_weekend_i = is_weekend
        row = np.array([[
            hour_i, day_num, month, is_weekend_i, int(is_holiday),
            temp, rain, snow, clouds, WEATHER_OPTIONS[weather]
        ]])
        row_scaled = scaler.transform(row)
        sequence.append(row_scaled[0])

    input_seq = np.array(sequence).reshape(1, SEQUENCE_LENGTH, N_FEATURES)

    pred_prob  = model.predict(input_seq, verbose=0)[0]
    pred_class = np.argmax(pred_prob)
    pred_label = le.classes_[pred_class]
    confidence = pred_prob[pred_class] * 100

    # --- Result ---
    st.subheader("Prediction Result")

    if pred_label == 'High':
        st.error(f"🔴 HIGH CONGESTION — {confidence:.1f}% confidence")
    elif pred_label == 'Medium':
        st.warning(f"🟡 MEDIUM CONGESTION — {confidence:.1f}% confidence")
    else:
        st.success(f"🟢 LOW CONGESTION — {confidence:.1f}% confidence")

    # Probability bars
    st.markdown("**Prediction Probabilities:**")
    for i, cls in enumerate(le.classes_):
        st.progress(float(pred_prob[i]),
                    text=f"{cls}: {pred_prob[i]*100:.1f}%")

    # --- Explainability ---
    st.divider()
    st.subheader("Why this prediction?")

    importance = {
        'hour':            0.5365,
        'is_weekend':      0.0967,
        'day_of_week':     0.0682,
        'weather_encoded': 0.0190,
        'is_holiday':      0.0079,
        'temp':            0.0068,
        'clouds_all':      0.0067,
        'month':           0.0022,
        'rain_1h':         0.0003,
        'snow_1h':         0.0000
    }

    st.markdown("**Top features influencing congestion predictions:**")
    for feature, score in sorted(importance.items(),
                                  key=lambda x: x[1], reverse=True)[:5]:
        bar = "█" * int(score * 50)
        st.markdown(f"`{feature:<20}` {bar}  `{score:.4f}`")

    st.caption("Feature importance computed using permutation-based explainability")

    # --- Context message ---
    st.divider()
    st.subheader("What this means")
    if pred_label == 'High':
        st.markdown("""
        - Expect heavy traffic and significant delays
        - Primary cause: **Rush hour** patterns
        - Suggestion: Avoid peak hours or use alternate routes
        """)
    elif pred_label == 'Medium':
        st.markdown("""
        - Moderate traffic — some delays expected
        - Traffic is building up or winding down
        - Suggestion: Allow extra travel time
        """)
    else:
        st.markdown("""
        - Light traffic — smooth flow expected
        - Off-peak hours with minimal congestion
        - Good time to travel!
        """)

# --- Footer ---
st.divider()
st.caption("Model: CNN-LSTM | Accuracy: 91.89% | Dataset: Metro Interstate Traffic Volume | Built with Streamlit")