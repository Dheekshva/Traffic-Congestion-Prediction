import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

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

# Bengaluru congestion pattern by hour (based on TomTom data)
HOURLY_PATTERN = {
    0: 0.12, 1: 0.08, 2: 0.06, 3: 0.05, 4: 0.07, 5: 0.15,
    6: 0.42, 7: 0.78, 8: 0.95, 9: 0.88, 10: 0.65, 11: 0.55,
    12: 0.60, 13: 0.58, 14: 0.62, 15: 0.70, 16: 0.85, 17: 0.98,
    18: 1.00, 19: 0.95, 20: 0.80, 21: 0.60, 22: 0.38, 23: 0.22
}

# Location congestion index
LOCATION_INDEX = {
    'Silk Board Junction':  0.98,
    'MG Road':              0.82,
    'Hebbal Flyover':       0.75,
    'Electronic City Toll': 0.70,
    'Marathahalli Bridge':  0.78,
    'Outer Ring Road':      0.85,
    'Whitefield':           0.65,
    'Koramangala':          0.80
}

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
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights('blr_weights.weights.h5')
    with open('scaler_blr.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('le_blr.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, scaler, le

model, scaler, le = load_model()

st.set_page_config(page_title="BLR Traffic Predictor", layout="wide")

# ── Styles ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background: #0c0f1a;
    color: #ddd8cc;
}
.main { background: #0c0f1a; }
.block-container { padding: 2rem 2rem 4rem; max-width: 1300px; }

.hero { padding: 2.5rem 0 1.5rem; }
.hero-tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #3d6b8a;
    margin-bottom: 0.8rem;
}
.hero-title {
    font-size: clamp(1.8rem, 4vw, 3rem);
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #ddd8cc;
    margin: 0 0 0.4rem;
    line-height: 1.1;
}
.hero-title b { color: #4a8fb5; }
.hero-meta {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #3a4a5a;
    letter-spacing: 0.05em;
}

.section-lbl {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #3a4a5a;
    margin: 2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}

.glass {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1.4rem;
    margin-bottom: 0.8rem;
}

.result-high   { background: rgba(160,30,30,0.15); border: 1px solid rgba(200,60,60,0.25); border-radius: 14px; padding: 1.8rem; }
.result-medium { background: rgba(160,100,20,0.15); border: 1px solid rgba(200,140,40,0.25); border-radius: 14px; padding: 1.8rem; }
.result-low    { background: rgba(20,130,70,0.15); border: 1px solid rgba(40,170,90,0.25); border-radius: 14px; padding: 1.8rem; }

.result-title  { font-size: 1.6rem; font-weight: 800; letter-spacing: -0.03em; margin: 0.3rem 0; color: #ddd8cc; }
.result-sub    { font-family: 'Space Mono', monospace; font-size: 0.65rem; color: #3a4a5a; letter-spacing: 0.05em; }

.badge-high   { display:inline-block; background:rgba(200,60,60,0.2); color:#f87171; border:1px solid rgba(200,60,60,0.3); border-radius:4px; padding:0.15rem 0.5rem; font-family:'Space Mono',monospace; font-size:0.6rem; letter-spacing:0.1em; text-transform:uppercase; }
.badge-medium { display:inline-block; background:rgba(200,140,40,0.2); color:#fbbf24; border:1px solid rgba(200,140,40,0.3); border-radius:4px; padding:0.15rem 0.5rem; font-family:'Space Mono',monospace; font-size:0.6rem; letter-spacing:0.1em; text-transform:uppercase; }
.badge-low    { display:inline-block; background:rgba(40,170,90,0.2); color:#34d399; border:1px solid rgba(40,170,90,0.3); border-radius:4px; padding:0.15rem 0.5rem; font-family:'Space Mono',monospace; font-size:0.6rem; letter-spacing:0.1em; text-transform:uppercase; }

.stat-grid { display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; margin-top:0.8rem; }
.stat-box { background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); border-radius:8px; padding:0.7rem 0.9rem; }
.stat-lbl { font-family:'Space Mono',monospace; font-size:0.58rem; letter-spacing:0.12em; text-transform:uppercase; color:#3a4a5a; margin-bottom:0.2rem; }
.stat-val { font-size:0.88rem; font-weight:700; color:#ddd8cc; }

.warn { background:rgba(160,110,20,0.1); border:1px solid rgba(200,150,40,0.2); border-radius:8px; padding:0.7rem 1rem; font-family:'Space Mono',monospace; font-size:0.68rem; color:#c8960a; margin-bottom:0.7rem; line-height:1.5; }

label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #3a4a5a !important;
}

.stButton > button {
    background: #1a3a52 !important;
    color: #7ec8e3 !important;
    border: 1px solid #2a5a7a !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.65rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #2a5a7a !important;
    border-color: #4a8fb5 !important;
}

footer, header, #MainMenu { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-tag">CNN-LSTM · Bengaluru Augmented · v2.0</p>
    <h1 class="hero-title">Bengaluru <b>Traffic</b> Congestion Predictor</h1>
    <p class="hero-meta">89.13% test accuracy · permutation-based explainability · TomTom Traffic Index 2022/2023</p>
</div>
""", unsafe_allow_html=True)

# ── Layout ──────────────────────────────────────────────────────────
col_in, col_out = st.columns([1, 1.2], gap="large")

with col_in:
    st.markdown('<div class="section-lbl">Input conditions</div>', unsafe_allow_html=True)
    location    = st.selectbox("Location", LOCATIONS)
    day_of_week = st.selectbox("Day of week", ['Monday','Tuesday','Wednesday',
                                                'Thursday','Friday','Saturday','Sunday'])
    hour        = st.slider("Hour of day", 0, 23, 8)
    month       = st.slider("Month", 1, 12, 6)
    temp        = st.slider("Temperature (C)", 10, 45, 28)
    weather     = st.selectbox("Weather", list(WEATHER_OPTIONS.keys()))
    rain        = st.slider("Rainfall (mm)", 0.0, 50.0, 0.0)
    clouds      = st.slider("Cloud cover (%)", 0, 100, 40)
    is_holiday  = st.toggle("Public holiday", value=False)

    day_num    = ['Monday','Tuesday','Wednesday',
                  'Thursday','Friday','Saturday','Sunday'].index(day_of_week)
    is_weekend = 1 if day_of_week in ['Saturday','Sunday'] else 0
    snow       = 0.0

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Run Prediction", use_container_width=True)

with col_out:
    st.markdown('<div class="section-lbl">Prediction output</div>', unsafe_allow_html=True)

    if predict_btn:

        # Warnings
        if month in MONSOON_MONTHS and rain > 0:
            st.markdown('<div class="warn">Monsoon conditions detected — Bengaluru roads typically experience 3 to 4x longer travel times during rainfall, particularly at Silk Board Junction and Outer Ring Road.</div>',
                        unsafe_allow_html=True)
        if day_of_week == 'Friday' and 18 <= hour <= 19:
            st.markdown('<div class="warn">Friday 6 to 7 PM is the historically worst congestion window in Bengaluru per TomTom Traffic Index 2022.</div>',
                        unsafe_allow_html=True)

        # Build sequence
        input_row    = np.array([[hour, day_num, month, is_weekend, int(is_holiday),
                                   temp, rain, snow, clouds, WEATHER_OPTIONS[weather]]])
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
        css   = {'High': 'result-high',   'Medium': 'result-medium',   'Low': 'result-low'}
        badge = {'High': 'badge-high',    'Medium': 'badge-medium',    'Low': 'badge-low'}
        lbl   = {'High': 'High Congestion','Medium': 'Moderate Congestion','Low': 'Low Congestion'}
        delay = {'High': '+25 to 45 min', 'Medium': '+10 to 20 min',   'Low': 'Minimal'}
        speed = {'High': '< 18 km/hr',   'Medium': '25 to 35 km/hr',  'Low': '> 40 km/hr'}
        act   = {'High': 'Use Metro or delay travel',
                 'Medium': 'Allow extra travel time',
                 'Low': 'Good conditions to travel'}

        st.markdown(f"""
        <div class="{css[pred_label]}">
            <span class="{badge[pred_label]}">{pred_label}</span>
            <p class="result-title">{lbl[pred_label]}</p>
            <p class="result-sub">{location} &nbsp;·&nbsp; {confidence:.1f}% confidence &nbsp;·&nbsp; {hour:02d}:00 hrs</p>
            <div class="stat-grid">
                <div class="stat-box"><div class="stat-lbl">Est. delay</div><div class="stat-val">{delay[pred_label]}</div></div>
                <div class="stat-box"><div class="stat-lbl">Avg speed</div><div class="stat-val">{speed[pred_label]}</div></div>
                <div class="stat-box" style="grid-column:span 2"><div class="stat-lbl">Suggestion</div><div class="stat-val">{act[pred_label]}</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── VISUALIZATION 1: Congestion Probability Gauge ──────────
        st.markdown('<div class="section-lbl">Congestion probability</div>', unsafe_allow_html=True)

        fig_gauge, ax = plt.subplots(figsize=(7, 2.2))
        fig_gauge.patch.set_facecolor('#0c0f1a')
        ax.set_facecolor('#0c0f1a')

        classes    = list(le.classes_)
        probs      = [pred_prob[i] * 100 for i in range(len(classes))]
        bar_colors = {'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'}
        colors     = [bar_colors[c] for c in classes]

        bars = ax.barh(classes, probs, color=colors, height=0.45,
                       alpha=0.85, zorder=3)
        ax.barh(classes, [100]*len(classes), color='#1a1f2e', height=0.45,
                alpha=1.0, zorder=2)
        bars = ax.barh(classes, probs, color=colors, height=0.45,
                       alpha=0.9, zorder=3)

        for bar, prob, cls in zip(bars, probs, classes):
            ax.text(min(prob + 1.5, 95), bar.get_y() + bar.get_height()/2,
                    f'{prob:.1f}%',
                    va='center', ha='left',
                    color=bar_colors[cls], fontsize=8,
                    fontfamily='monospace', fontweight='bold')

        ax.set_xlim(0, 105)
        ax.set_xlabel('Probability (%)', color='#3a4a5a', fontsize=7, fontfamily='monospace')
        ax.tick_params(colors='#3a4a5a', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#1a2030')
        ax.yaxis.label.set_color('#3a4a5a')
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes, color='#ddd8cc', fontsize=8, fontfamily='monospace')
        ax.xaxis.set_tick_params(colors='#3a4a5a')
        plt.tight_layout(pad=0.5)
        st.pyplot(fig_gauge, use_container_width=True)
        plt.close()

        # ── VISUALIZATION 2: Hourly Congestion Heatmap ─────────────
        st.markdown('<div class="section-lbl">Hourly congestion pattern — Bengaluru</div>',
                    unsafe_allow_html=True)

        fig_heat, ax2 = plt.subplots(figsize=(7, 1.8))
        fig_heat.patch.set_facecolor('#0c0f1a')
        ax2.set_facecolor('#0c0f1a')

        hours  = list(range(24))
        values = [HOURLY_PATTERN[h] for h in hours]

        # Custom colormap — green to amber to red
        cmap = LinearSegmentedColormap.from_list(
            'traffic', ['#10b981', '#f59e0b', '#ef4444'], N=256)

        for i, (h, v) in enumerate(zip(hours, values)):
            color = cmap(v)
            ax2.bar(h, 1, color=color, width=0.92, alpha=0.9)
            if h == hour:
                ax2.bar(h, 1.15, color='white', width=0.92, alpha=0.15)
                ax2.axvline(h, color='white', linewidth=1, alpha=0.5, linestyle='--')

        ax2.set_xlim(-0.5, 23.5)
        ax2.set_ylim(0, 1.3)
        ax2.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 23])
        ax2.set_xticklabels(['12am','3am','6am','9am','12pm','3pm','6pm','9pm','11pm'],
                             color='#3a4a5a', fontsize=7, fontfamily='monospace')
        ax2.set_yticks([])
        for spine in ax2.spines.values():
            spine.set_visible(False)

        ax2.text(hour, 1.22, f'{hour:02d}:00', ha='center', va='bottom',
                 color='white', fontsize=7, fontfamily='monospace')

        # Legend
        for label_txt, color_hex, xpos in [('Low', '#10b981', 0.72),
                                             ('Medium', '#f59e0b', 0.82),
                                             ('High', '#ef4444', 0.92)]:
            ax2.add_patch(mpatches.FancyBboxPatch(
                (xpos * 24 - 0.5, 1.08), 0.4, 0.12,
                boxstyle='round,pad=0.05',
                facecolor=color_hex, alpha=0.8, transform=ax2.transData))
            ax2.text(xpos * 24 + 0.2, 1.14, label_txt,
                     color='white', fontsize=6, va='center',
                     fontfamily='monospace', transform=ax2.transData)

        plt.tight_layout(pad=0.3)
        st.pyplot(fig_heat, use_container_width=True)
        plt.close()

        # ── VISUALIZATION 3: Location Comparison ───────────────────
        st.markdown('<div class="section-lbl">Congestion index by location</div>',
                    unsafe_allow_html=True)

        fig_loc, ax3 = plt.subplots(figsize=(7, 2.8))
        fig_loc.patch.set_facecolor('#0c0f1a')
        ax3.set_facecolor('#0c0f1a')

        locs   = list(LOCATION_INDEX.keys())
        vals   = [LOCATION_INDEX[l] * HOURLY_PATTERN[hour] for l in locs]
        short  = [l.split()[0] + '\n' + ' '.join(l.split()[1:]) if len(l.split()) > 1
                  else l for l in locs]
        bar_c  = [cmap(v) for v in vals]

        bars3 = ax3.bar(range(len(locs)), vals, color=bar_c, width=0.6, alpha=0.9)

        # Highlight selected location
        sel_idx = locs.index(location)
        ax3.bar(sel_idx, vals[sel_idx], color='white', width=0.6, alpha=0.12)
        ax3.bar(sel_idx, vals[sel_idx], color=bar_c[sel_idx],
                width=0.6, alpha=0.9, edgecolor='white', linewidth=1.5)

        ax3.set_xticks(range(len(locs)))
        ax3.set_xticklabels(short, color='#3a4a5a', fontsize=6.5,
                             fontfamily='monospace', ha='center')
        ax3.set_ylim(0, 1.15)
        ax3.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax3.set_yticklabels(['0', '0.25', '0.50', '0.75', '1.0'],
                             color='#3a4a5a', fontsize=6.5, fontfamily='monospace')
        ax3.set_ylabel('Congestion Index', color='#3a4a5a',
                       fontsize=7, fontfamily='monospace')
        ax3.axhline(0.75, color='#f59e0b', linewidth=0.5, linestyle='--', alpha=0.5)
        ax3.axhline(0.45, color='#10b981', linewidth=0.5, linestyle='--', alpha=0.5)

        for spine in ax3.spines.values():
            spine.set_color('#1a2030')
        ax3.tick_params(colors='#3a4a5a')

        ax3.text(len(locs) - 0.4, 0.76, 'High threshold',
                 color='#f59e0b', fontsize=6, fontfamily='monospace', va='bottom')
        ax3.text(len(locs) - 0.4, 0.46, 'Med threshold',
                 color='#10b981', fontsize=6, fontfamily='monospace', va='bottom')

        plt.tight_layout(pad=0.4)
        st.pyplot(fig_loc, use_container_width=True)
        plt.close()

        # ── VISUALIZATION 4: Feature Importance ────────────────────
        st.markdown('<div class="section-lbl">Feature influence (permutation importance)</div>',
                    unsafe_allow_html=True)

        importance = [
            ('hour',        0.5365),
            ('is_weekend',  0.0967),
            ('day_of_week', 0.0682),
            ('weather',     0.0190),
            ('is_holiday',  0.0079),
            ('temperature', 0.0068),
            ('cloud_cover', 0.0067),
            ('month',       0.0022),
            ('rainfall',    0.0003),
        ]

        feat_names = [f[0] for f in importance]
        feat_vals  = [f[1] for f in importance]
        max_v      = feat_vals[0]
        norm_vals  = [v / max_v for v in feat_vals]

        fig_feat, ax4 = plt.subplots(figsize=(7, 2.8))
        fig_feat.patch.set_facecolor('#0c0f1a')
        ax4.set_facecolor('#0c0f1a')

        feat_colors = ['#4a8fb5' if v > 0.3 else '#2d6a8f' if v > 0.1
                       else '#1a3a52' for v in norm_vals]

        bars4 = ax4.barh(feat_names[::-1], feat_vals[::-1],
                          color=feat_colors[::-1], height=0.55, alpha=0.9)

        for bar, val in zip(bars4, feat_vals[::-1]):
            ax4.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                     f'{val:.4f}', va='center', color='#3a4a5a',
                     fontsize=6.5, fontfamily='monospace')

        ax4.set_xlim(0, 0.62)
        ax4.set_xlabel('Accuracy drop when feature shuffled',
                       color='#3a4a5a', fontsize=7, fontfamily='monospace')
        ax4.tick_params(colors='#3a4a5a', labelsize=7.5)
        for spine in ax4.spines.values():
            spine.set_color('#1a2030')
        ax4.set_yticklabels(feat_names[::-1], color='#ddd8cc',
                             fontsize=7.5, fontfamily='monospace')
        ax4.xaxis.set_tick_params(colors='#3a4a5a')
        plt.tight_layout(pad=0.4)
        st.pyplot(fig_feat, use_container_width=True)
        plt.close()

        st.markdown("""
        <p style="font-family:'Space Mono',monospace; font-size:0.58rem; color:#2d3748; margin-top:0.5rem; letter-spacing:0.05em">
        Permutation importance · TomTom Traffic Index 2022/2023 · CNN-LSTM model · 89.13% accuracy
        </p>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="padding:5rem 2rem; text-align:center; color:#2d3748;
                    font-family:'Space Mono',monospace; font-size:0.72rem; letter-spacing:0.05em">
            Set conditions on the left and run prediction.
        </div>
        """, unsafe_allow_html=True)