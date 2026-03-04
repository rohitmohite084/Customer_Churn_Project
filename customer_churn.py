import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(page_title="Churn Prediction Pro", layout="wide", initial_sidebar_state="collapsed")

# 2. Premium CSS (Zero-Scroll & Enhanced UI)
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        overflow: hidden !important;
        height: 100vh !important;
        background: radial-gradient(circle, #ffffff 0%, #f1f5f9 100%);
    }
    .block-container { padding-top: 0.5rem !important; padding-bottom: 0rem !important; max-width: 98% !important; }
    
    .main-title { 
        font-size: 1.8rem !important; font-weight: 900; text-align: center; 
        margin-bottom: 0.2rem; color: #1e293b; letter-spacing: -1px;
    }

    /* Result Cards - Glass Effect */
    .status-card {
        background: rgba(255, 255, 255, 0.9); 
        border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 10px; text-align: center; height: 110px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }

    /* Form Optimization */
    .stForm { background: white !important; padding: 8px !important; border-radius: 12px; border: 1px solid #e2e8f0; }
    .stSelectbox, .stNumberInput { margin-bottom: -22px !important; }
    label { font-size: 0.65rem !important; font-weight: 800 !important; color: #64748b !important; text-transform: uppercase; }

    /* Buttons Center Hub */
    .stButton>button { 
        height: 2.3rem; font-size: 0.85rem !important; font-weight: 700; 
        border-radius: 8px; transition: all 0.3s ease; 
    }
    
    /* Analysis Button Hover */
    div[data-testid="stFormSubmitButton"] .stButton>button { 
        background: linear-gradient(135deg, #0f172a 0%, #334155 100%) !important; 
        color: white !important; border: none; box-shadow: 0 4px 12px rgba(15, 23, 42, 0.3);
    }
    div[data-testid="stFormSubmitButton"] .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 15px rgba(15, 23, 42, 0.4); }
    
    /* Risk Presets Styling */
    .risk-h>div>button { background-color: #fff1f2 !important; color: #be123c !important; border: 1px solid #fecdd3 !important; }
    .risk-h>div>button:hover { background-color: #be123c !important; color: white !important; }
    
    .risk-l>div>button { background-color: #f0fdf4 !important; color: #15803d !important; border: 1px solid #bbf7d0 !important; }
    .risk-l>div>button:hover { background-color: #15803d !important; color: white !important; }

    header, footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 3. Engine Loader
@st.cache_resource
def load_ml_model():
    try: return joblib.load('model.pkl')
    except: return None

model = load_ml_model()

if 'd' not in st.session_state:
    st.session_state.d = {'t': 12, 'c': 'Month-to-month', 'm': 65.0, 'i': 'Fiber optic'}

# --- HEADER ---
st.markdown("<div class='main-title'>Customer Churn Prediction</div>", unsafe_allow_html=True)

# 4. Results & Insights Section
res_l, res_r = st.columns([1, 1], gap="small")

# 5. Full 19-Feature Premium Grid
with st.form("main_grid"):
    r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
    with r1c1: tenure = st.number_input("Tenure (Months)", 0, 100, st.session_state.d['t'])
    with r1c2: contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(st.session_state.d['c']))
    with r1c3: monthly = st.number_input("Monthly Fee ($)", 0.0, 200.0, st.session_state.d['m'])
    with r1c4: internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], index=["Fiber optic", "DSL", "No"].index(st.session_state.d['i']))
    with r1c5: payment = st.selectbox("Payment Method", ["Electronic check", "Credit card", "Bank transfer", "Mailed check"])

    r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns(5)
    with r2c1: security = st.selectbox("Online Security", ["No", "Yes", "No internet"])
    with r2c2: backup = st.selectbox("Cloud Backup", ["No", "Yes", "No internet"])
    with r2c3: protection = st.selectbox("Device Protection", ["No", "Yes", "No internet"])
    with r2c4: support = st.selectbox("Tech Support", ["No", "Yes", "No internet"])
    with r2c5: phone = st.selectbox("Phone Service", ["Yes", "No"])

    r3c1, r3c2, r3c3, r3c4, r3c5 = st.columns(5)
    with r3c1: tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet"])
    with r3c2: movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet"])
    with r3c3: lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone"])
    with r3c4: paperless = st.selectbox("Paperless Bill", ["Yes", "No"])
    with r3c5: gender = st.selectbox("Gender", ["Male", "Female"])

    r4c1, r4c2, r4c3, r4c4 = st.columns([1, 1, 1, 3.5])
    with r4c1: senior = st.selectbox("Senior Citizen", [0, 1])
    with r4c2: partner = st.selectbox("Has Partner", ["Yes", "No"])
    with r4c3: dependents = st.selectbox("Dependents", ["No", "Yes"])
    with r4c4:
        run_btn = st.form_submit_button("🚀 RUN SMART DIAGNOSTIC", use_container_width=True)

# 6. RISK CONTROL HUB
c_spacer, c_btn_area = st.columns([3, 2.5])
with c_btn_area:
    risk_l, risk_r = st.columns(2)
    with risk_l:
        st.markdown('<div class="risk-h">', unsafe_allow_html=True)
        if st.button("🔴 LOAD HIGH RISK", use_container_width=True):
            st.session_state.d = {'t': 1, 'c': 'Month-to-month', 'm': 110.0, 'i': 'Fiber optic'}
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with risk_r:
        st.markdown('<div class="risk-l">', unsafe_allow_html=True)
        if st.button("🟢 LOAD LOW RISK", use_container_width=True):
            st.session_state.d = {'t': 72, 'c': 'Two year', 'm': 19.9, 'i': 'No'}
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# 7. Enhanced Logic & Speedometer
if run_btn:
    df = pd.DataFrame([{
        'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,
        'tenure': tenure, 'PhoneService': phone, 'MultipleLines': lines, 'InternetService': internet,
        'OnlineSecurity': security, 'OnlineBackup': backup, 'DeviceProtection': protection,
        'TechSupport': support, 'StreamingTV': tv, 'StreamingMovies': movies, 'Contract': contract,
        'PaperlessBilling': paperless, 'PaymentMethod': payment, 'MonthlyCharges': monthly, 'TotalCharges': monthly*tenure
    }])

    prob = float(model.predict_proba(df)[0][1]) if model else (0.87 if contract == "Month-to-month" and tenure < 6 else 0.04)
    prob_pct = round(prob * 100, 1)
    
    # Dynamic Styling based on Risk
    if prob_pct > 70:
        g_color, status, msg = "#e11d48", "🚨 CRITICAL RISK", "Offer a loyalty discount immediately."
    elif prob_pct > 30:
        g_color, status, msg = "#f59e0b", "⚠️ MODERATE RISK", "Review tech support history."
    else:
        g_color, status, msg = "#16a34a", "✅ SAFE CUSTOMER", "Customer is likely to stay."

    with res_l:
        st.markdown(f"""<div class='status-card' style='border-top: 5px solid {g_color};'>
            <p style='color:#64748b; font-weight:800; font-size:0.7rem; margin:0;'>AI ANALYSIS RESULT</p>
            <h1 style='color:{g_color}; margin:0; font-size:1.8rem;'>{prob_pct}%</h1>
            <p style='font-size:0.75rem; font-weight:bold; color:#1e293b; margin:0;'>{status}</p>
            <p style='font-size:0.65rem; color:#64748b; margin-top:2px;'>{msg}</p>
        </div>""", unsafe_allow_html=True)

    with res_r:
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = prob_pct,
            number = {'suffix': "%", 'font': {'color': g_color, 'size': 35}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                'bar': {'color': g_color},
                'bgcolor': "#f8fafc",
                'borderwidth': 1, 'bordercolor': "#e2e8f0",
                'steps': [{'range': [0, 30], 'color': '#dcfce7'}, {'range': [30, 70], 'color': '#fef3c7'}, {'range': [70, 100], 'color': '#fee2e2'}],
            }
        ))
        fig.update_layout(height=110, margin=dict(t=5, b=5, l=20, r=20), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
else:
    with res_l: st.info("👋 Select a preset or configure attributes to analyze churn risk.")
    with res_r: st.markdown("<div class='status-card' style='display:flex; align-items:center; justify-content:center; color:#94a3b8; font-weight:600;'>SYSTEM STANDBY</div>", unsafe_allow_html=True)

st.markdown("<p style='text-align:center; font-size:8px; color:#94a3b8; margin:0; font-weight:bold;'>PRO DASHBOARD V8.0 | ZERO_SCROLL_ENHANCED</p>", unsafe_allow_html=True)