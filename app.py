import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils import predict_co2, get_categorical_options, load_recommender

# 1. Page Configuration
st.set_page_config(page_title="CO2 Emission Predictor", page_icon="🍃", layout="wide")

# Inject Custom CSS for modern dark-mode dashboard styling with glassmorphism and glow effects
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

/* Main font and styling */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Glassmorphism Card Container (native Streamlit container with border) */
div[data-testid="stContainerBorder"] {
    background-color: rgba(30, 30, 47, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    padding: 24px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
    backdrop-filter: blur(8px) !important;
    -webkit-backdrop-filter: blur(8px) !important;
    margin-bottom: 20px !important;
}

/* Prediction Output Card with Green Glow */
.prediction-card {
    background: linear-gradient(135deg, rgba(0, 192, 75, 0.1), rgba(0, 229, 255, 0.05));
    border: 1px solid rgba(0, 192, 75, 0.3);
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 10px 30px rgba(0, 192, 75, 0.15);
    text-align: center;
    margin-bottom: 20px;
}

/* Sidebar Styling */
.stSidebar {
    background-color: #0e0e15 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Glow on primary submit buttons */
button[kind="primary"] {
    background: linear-gradient(45deg, #00c04b, #00e5ff) !important;
    color: #0e0e15 !important;
    font-weight: 700 !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(0, 192, 75, 0.3) !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    padding: 12px 0 !important;
    border-radius: 8px !important;
}
button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(0, 192, 75, 0.5) !important;
    transform: translateY(-2px);
}

/* Glow on secondary buttons */
button[kind="secondary"] {
    border: 1px solid rgba(0, 192, 75, 0.5) !important;
    background-color: transparent !important;
    color: #00c04b !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    border-radius: 8px !important;
}
button[kind="secondary"]:hover {
    background-color: rgba(0, 192, 75, 0.1) !important;
    color: #00e5ff !important;
    border-color: #00e5ff !important;
}

/* Section Dividers */
hr {
    border: 0;
    height: 1px;
    background: linear-gradient(to right, rgba(0, 192, 75, 0), rgba(0, 192, 75, 0.5), rgba(0, 192, 75, 0));
    margin: 30px 0;
}
</style>
""", unsafe_allow_html=True)

# 2. Main Title with Gradient
st.markdown("""
<div style='text-align: center; padding: 20px 0;'>
    <h1 style='font-size: 2.8em; font-weight: 800; background: -webkit-linear-gradient(45deg, #00c04b, #00e5ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 5px;'>
        CO₂ Emission Intelligence Predictor 🍃
    </h1>
    <p style='color: #a0a0b0; font-size: 1.2em; font-weight: 300;'>
        Predict passenger vehicle emissions using hyper-tuned XGBoost machine learning.
    </p>
</div>
""", unsafe_allow_html=True)

# Load options for dropdowns dynamically
try:
    options = get_categorical_options()
    ref_payload = load_recommender()
    ref_df = ref_payload['reference_df']
except Exception as e:
    st.error(f"Error loading model resources: {e}. Please ensure you ran all notebooks and fit the recommender.")
    st.stop()

# 3. Main Dashboard Layout (Columns)
col_input, col_result = st.columns([1, 1.2], gap="large")

with col_input:
    with st.container(border=True):
        st.markdown("<h3 style='margin-top:0; color:#00e5ff; font-weight:600;'>🚗 Vehicle Specifications</h3>", unsafe_allow_html=True)
        st.write("Fill in the specs below to run the ML model:")
    
    # Form for inputs
    with st.form("specs_form"):
        # Selectbox for Make (pre-filled with Acura)
        make = st.selectbox("Vehicle Make", options=options['Make'], index=options['Make'].index("ACURA") if "ACURA" in options['Make'] else 0)
        
        # Selectbox for Vehicle Class (pre-filled with COMPACT)
        vehicle_class = st.selectbox("Vehicle Class", options=options['Vehicle class'], index=options['Vehicle class'].index("COMPACT") if "COMPACT" in options['Vehicle class'] else 0)
        
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            fuel_type = st.selectbox(
                "Fuel Type", 
                options=options['Fuel type'],
                format_func=lambda x: {
                    "X": "Regular Gasoline", 
                    "Z": "Premium Gasoline", 
                    "D": "Diesel", 
                    "E": "Ethanol (E85)"
                }.get(x, x)
            )
        with sub_col2:
            transmission = st.selectbox(
                "Transmission Type", 
                options=options['Transmission type'],
                format_func=lambda x: {
                    "A": "Automatic", 
                    "AM": "Automated Manual", 
                    "AS": "Automatic with Select Shift", 
                    "AV": "Continuously Variable (CVT)", 
                    "M": "Manual"
                }.get(x, x)
            )
            
        sub_col3, sub_col4, sub_col5 = st.columns(3)
        with sub_col3:
            engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
        with sub_col4:
            gears = st.number_input("Gears", min_value=0, max_value=10, value=6, step=1)
        with sub_col5:
            model_year = st.number_input("Model Year", min_value=2015, max_value=2025, value=2020, step=1)
            
        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("Predict Carbon Footprint", type="primary")

    # 4. Extreme Bounds Guardrails
    if engine_size > 6.0 and gears < 4 and gears > 0:
        st.warning("⚠️ **Unusual Specification Combo:** A massive engine (>6.0L) with few gears (<4) is extremely rare. Model variance may be higher.")
    elif engine_size < 1.0 and gears > 8:
        st.warning("⚠️ **Unusual Specification Combo:** A small engine (<1.0L) with 9+ gears is highly atypical. Model predictions could be less reliable.")

with col_result:
    if submit_btn or 'prediction' in st.session_state:
        # If user just submitted, compute and save
        if submit_btn:
            input_dict = {
                'Model year': model_year,
                'Make': make,
                'Vehicle class': vehicle_class,
                'Engine size (L)': engine_size,
                'Fuel type': fuel_type,
                'Transmission type': transmission,
                'Gears': gears
            }
            
            try:
                prediction = predict_co2(input_dict)
                st.session_state['prediction'] = prediction
                st.session_state['input_specs'] = input_dict
            except Exception as e:
                st.error(f"Error executing prediction: {e}")
                st.stop()
        else:
            prediction = st.session_state['prediction']
            input_dict = st.session_state['input_specs']
            # Update inputs to match session state if loading page
            model_year = input_dict['Model year']
            make = input_dict['Make']
            vehicle_class = input_dict['Vehicle class']
            fuel_type = input_dict['Fuel type']
            transmission = input_dict['Transmission type']
            engine_size = input_dict['Engine size (L)']
            gears = input_dict['Gears']

        mae = 9.10 # MAE of tuned XGBoost on test set

        # Output Card
        st.markdown(f"""
        <div class="prediction-card">
            <h4 style="margin: 0; color: #a0a0b0; text-transform: uppercase; font-size: 0.9em; letter-spacing: 1px;">Estimated CO₂ Emissions</h4>
            <h1 style="margin: 10px 0; font-size: 3.5em; font-weight: 800; color: #00c04b; text-shadow: 0 0 15px rgba(0, 192, 75, 0.3);">{prediction:.0f} <span style="font-size: 0.4em; color: white; font-weight: 400;">g/km</span></h1>
            <p style="margin: 0; color: #a0a0b0;">Typical prediction error: <b>± {mae:.1f} g/km</b> (Model MAE)</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Row of Buttons below card
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🍃 Find Greener Alternatives", type="secondary", use_container_width=True):
                st.switch_page("pages/2_Recommend.py")
        with btn_col2:
            st.info("💡 Switch to page 2 in the sidebar to view similar low-emission models.")

        # Plotly Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Emissions Zone Relative to Canada Dataset", 'font': {'size': 15, 'color': '#a0a0b0', 'family': 'Outfit'}},
            number = {'font': {'color': 'white', 'family': 'Outfit', 'size': 1}}, # hide text, show on card
            gauge = {
                'axis': {'range': [50, 500], 'tickcolor': "#a0a0b0", 'tickwidth': 1, 'tickfont': {'color': '#a0a0b0'}},
                'bar': {'color': "rgba(0,0,0,0)"},
                'steps': [
                    {'range': [50, 180], 'color': "rgba(0, 192, 75, 0.2)"},    # Low (Green)
                    {'range': [180, 270], 'color': "rgba(255, 183, 3, 0.2)"},  # Medium (Amber)
                    {'range': [270, 500], 'color': "rgba(239, 35, 60, 0.2)"}   # High (Red)
                ],
                'threshold': {
                    'line': {'color': "#00e5ff", 'width': 5}, # Cyan needle indicator
                    'thickness': 0.75,
                    'value': prediction
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=40, b=10),
            height=250
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    else:
        # Placeholder before user hits predict
        st.markdown("""
        <div style="background-color: rgba(30, 30, 47, 0.3); border: 1px dashed rgba(255, 255, 255, 0.1); border-radius: 12px; height: 380px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; padding: 40px;">
            <span style="font-size: 4em; filter: grayscale(50%);">🚗</span>
            <h3 style="color: #a0a0b0; font-weight: 500; margin-top: 15px;">Awaiting Vehicle Specs</h3>
            <p style="color: #606070; max-width: 300px; font-size: 0.95em;">Select parameters on the left and click <b>Predict Carbon Footprint</b> to calculate carbon levels.</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# 5. Collapsible Dashboard Section for Model Metrics & Dataset Analysis
with st.expander("📊 Data Science Dashboard — Model Metrics & Reference Distribution", expanded=False):
    st.markdown("### Model Comparison & Performance Metrics")
    st.write("We evaluated 6 regression models on the NRCan Dataset. Hyper-tuned XGBoost outperformed the other pipelines with the highest R² score and lowest error:")
    
    # Render Comparison Metrics Table
    metrics_data = {
        'Model': ['XGBoost (Tuned - Production)', 'Random Forest (Tuned)', 'Random Forest (Untuned)', 'Decision Tree', 'Linear Regression', 'Ridge Regression', 'SGD (Gradient Descent)'],
        'CV R² Score': [0.9382, 0.9343, 0.9303, 0.9155, 0.8404, 0.8400, 0.7776],
        'Test R² Score': [0.9509, 0.9499, 0.9475, 0.9357, 0.8460, 0.8456, 0.7937],
        'Test RMSE (g/km)': [13.29, 13.43, 13.75, 15.21, 23.55, 23.58, 27.26],
        'Test MAE (g/km)': [9.10, 9.20, 9.25, 9.71, 17.84, 17.88, 20.41],
    }
    metrics_df = pd.DataFrame(metrics_data)
    st.table(metrics_df.style.highlight_max(subset=['CV R² Score', 'Test R² Score'], color='rgba(0,192,75,0.2)').highlight_min(subset=['Test RMSE (g/km)', 'Test MAE (g/km)'], color='rgba(0,192,75,0.2)'))
    
    st.markdown("### Carbon Emissions Distribution Across Dataset")
    st.write("Below is the distribution of actual CO₂ emissions across the ~10,060 vehicles in the Natural Resources Canada reference dataset.")
    
    # Plotly Histogram of reference emissions
    fig_hist = px.histogram(
        ref_df, 
        x="CO2 emissions (g/km)", 
        nbins=50,
        title="CO₂ Emissions Distribution Histogram",
        labels={"CO2 emissions (g/km)": "Actual CO₂ Emissions (g/km)"},
        color_discrete_sequence=["#00c04b"]
    )
    
    # If prediction exists, add a vertical line indicating current prediction
    if 'prediction' in st.session_state:
        pred_val = st.session_state['prediction']
        fig_hist.add_vline(
            x=pred_val, 
            line_width=3, 
            line_dash="dash", 
            line_color="#00e5ff",
            annotation_text=f"Your Vehicle ({pred_val:.0f} g/km)", 
            annotation_position="top right"
        )
        
    fig_hist.update_layout(
        paper_bgcolor="rgba(30, 30, 47, 0.4)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit", color="#a0a0b0"),
        title_font=dict(color="#00e5ff"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=350,
        yaxis_title="Count of Vehicles",
        bargap=0.05
    )
    fig_hist.update_xaxes(showgrid=False, zeroline=False)
    fig_hist.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False)
    
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("""
    #### Methodology Summary
    1. **Data Cleaning**: Redundant and regulatory ratings dropped; expected combined values verified mathematically.
    2. **Feature Engineering**: Split transmission into Gears and Type; mapped high-cardinality elements. Excluded cylinders due to multicollinearity (high VIF) after empirical ablation CV trials.
    3. **Model Selection**: Evaluated various model classes. Tuned XGBoost outperformed other models on all held-out test indicators.
    """)