import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils import get_recommendations

# 1. Page Configuration
st.set_page_config(page_title="CO2 Recommender", page_icon="🍃", layout="wide")

# Inject Custom CSS for dark-mode dashboard styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Glassmorphic card styling (native Streamlit container with border) */
div[data-testid="stContainerBorder"] {
    background-color: rgba(30, 30, 47, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    padding: 24px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
    backdrop-filter: blur(8px) !important;
    margin-bottom: 20px !important;
}

/* Accent Card for User Specs */
.user-specs-card {
    background-color: rgba(14, 14, 21, 0.5);
    border-left: 5px solid #00e5ff;
    border-radius: 4px 12px 12px 4px;
    padding: 16px 20px;
    margin-bottom: 20px;
}

/* Green recommendation cards */
.rec-box {
    background: linear-gradient(135deg, rgba(0, 192, 75, 0.08), rgba(30, 30, 47, 0.6));
    border: 1px solid rgba(0, 192, 75, 0.2);
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 12px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.rec-box:hover {
    transform: translateY(-2px);
    border-color: rgba(0, 192, 75, 0.5);
    box-shadow: 0 4px 15px rgba(0, 192, 75, 0.1);
}

/* Glow on primary navigation buttons */
button[kind="primary"] {
    background: linear-gradient(45deg, #00c04b, #00e5ff) !important;
    color: #0e0e15 !important;
    font-weight: 700 !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(0, 192, 75, 0.3) !important;
    transition: all 0.3s ease !important;
    border-radius: 8px !important;
}
button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(0, 192, 75, 0.5) !important;
    transform: translateY(-2px);
}

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
        🍃 Eco-Friendly Vehicle Recommender
    </h1>
    <p style='color: #a0a0b0; font-size: 1.2em; font-weight: 300;'>
        Find real vehicles that hit lower emissions with similar technical specifications.
    </p>
</div>
""", unsafe_allow_html=True)

# Check if session state contains a prediction. If not, direct them to run a prediction first.
if 'prediction' not in st.session_state or 'input_specs' not in st.session_state:
    st.markdown("""
    <div style="background-color: rgba(30, 30, 47, 0.3); border: 1px dashed rgba(239, 35, 60, 0.3); border-radius: 12px; max-width: 600px; margin: 40px auto; text-align: center; padding: 40px;">
        <span style="font-size: 4em; filter: grayscale(30%);">🔍</span>
        <h3 style="color: #ef233c; font-weight: 600; margin-top: 15px;">No Prediction Active</h3>
        <p style="color: #a0a0b0; font-size: 1em; margin-bottom: 25px;">The recommender requires vehicle specifications to locate alternatives. Please run a prediction first to initialize the engine.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_c, col_btn, col_d = st.columns([1.5, 1, 1.5])
    with col_btn:
        if st.button("Go to Predictor Page", type="primary", use_container_width=True):
            st.switch_page("app.py")
    st.stop()

# Retrieve prediction specs and outputs from session state
prediction = st.session_state['prediction']
input_specs = st.session_state['input_specs']

# Mapping functions for readable labels
fuel_type_map = {"X": "Regular Gasoline", "Z": "Premium Gasoline", "D": "Diesel", "E": "Ethanol (E85)"}
trans_map = {"A": "Automatic", "AM": "Automated Manual", "AS": "Automatic with Select Shift", "AV": "CVT", "M": "Manual"}

# 3. Left-Right layout
col_left, col_right = st.columns([1, 1.3], gap="large")

with col_left:
    with st.container(border=True):
        st.markdown("<h3 style='margin-top:0; color:#00e5ff; font-weight:600;'>📋 Target Specifications</h3>", unsafe_allow_html=True)
    st.write("Alternatives will be matched against these specifications:")
    
    # Render user specs cards
    st.markdown(f"""
    <div class="user-specs-card">
        <span style="color: #a0a0b0; font-size: 0.85em; text-transform: uppercase;">Make / Class</span>
        <h4 style="margin: 2px 0; color: white;">{input_specs['Make']} ({input_specs['Vehicle class']})</h4>
    </div>
    <div class="user-specs-card">
        <span style="color: #a0a0b0; font-size: 0.85em; text-transform: uppercase;">Engine / Transmission / Gears</span>
        <h4 style="margin: 2px 0; color: white;">{input_specs['Engine size (L)']}L / {trans_map.get(input_specs['Transmission type'], input_specs['Transmission type'])} / {input_specs['Gears']} Gears</h4>
    </div>
    <div class="user-specs-card">
        <span style="color: #a0a0b0; font-size: 0.85em; text-transform: uppercase;">Fuel Type / Model Year</span>
        <h4 style="margin: 2px 0; color: white;">{fuel_type_map.get(input_specs['Fuel type'], input_specs['Fuel type'])} / {input_specs['Model year']}</h4>
    </div>
    <div class="user-specs-card" style="border-left-color: #00c04b;">
        <span style="color: #a0a0b0; font-size: 0.85em; text-transform: uppercase;">Predicted Emissions</span>
        <h3 style="margin: 2px 0; color: #00c04b; font-weight: 800;">{prediction:.0f} g/km</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Recommendation control
    st.write("---")
    k = st.slider("Number of recommendations to show", min_value=3, max_value=8, value=5, step=1)
    
    if st.button("🔄 Refresh Recommendations", type="secondary", use_container_width=True):
        st.rerun()

with col_right:
    with st.container(border=True):
        st.markdown("<h3 style='margin-top:0; color:#00c04b; font-weight:600;'>🍃 Recommended Lower-Emission Sibling Models</h3>", unsafe_allow_html=True)
    st.write("Nearest neighbors by cosine similarity, filtered to actual emissions less than predicted:")
    
    # Retrieve recommendations using our cached utils function
    try:
        recs = get_recommendations(input_specs, prediction, k=k)
    except Exception as e:
        st.error(f"Error fetching recommendations: {e}")
        st.stop()
        
    if recs.empty:
        st.info("No lower-emission alternatives found in the dataset with similar specs. (Your target vehicle might already represent the most eco-friendly profile for these specifications).")
    else:
        # Display recommendations
        for idx, row in recs.iterrows():
            emission_diff = prediction - row['CO2 emissions (g/km)']
            percentage_reduction = (emission_diff / prediction) * 100
            
            st.markdown(f"""
            <div class="rec-box">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h4 style="margin: 0; color: white; font-size: 1.15em;">{row['Make']} {row['Model']}</h4>
                        <p style="margin: 2px 0 0 0; color: #a0a0b0; font-size: 0.9em;">
                            Class: <b>{row['Vehicle class']}</b> | Specs: <b>{row['Engine size (L)']}L</b>, <b>{row['Gears']} Gears ({row['Transmission type']})</b>
                        </p>
                    </div>
                    <div style="text-align: right;">
                        <h4 style="margin: 0; color: #00c04b; font-size: 1.25em; font-weight: 800;">{row['CO2 emissions (g/km)']:.0f} g/km</h4>
                        <span style="color: #00e5ff; font-size: 0.85em; font-weight: 600;">-{percentage_reduction:.0f}% emissions</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# 4. Emissions Comparison Chart (underneath target and options)
if not recs.empty:
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.markdown("### 📊 Emission Savings Comparison")
    st.write("Visualizing your vehicle specs vs. the recommended models:")
    
    # Assemble comparison dataframe
    comparison_list = [{
        'Label': 'Your Selected Specs (XGBoost Prediction)',
        'CO2 Emissions (g/km)': prediction,
        'Type': 'Target'
    }]
    
    for idx, row in recs.iterrows():
        comparison_list.append({
            'Label': f"{row['Make']} {row['Model']} ({row['Engine size (L)']}L {row['Fuel type']})",
            'CO2 Emissions (g/km)': row['CO2 emissions (g/km)'],
            'Type': 'Alternative'
        })
        
    comp_df = pd.DataFrame(comparison_list)
    
    # Generate Plotly Bar Chart
    fig_bar = px.bar(
        comp_df,
        y='Label',
        x='CO2 Emissions (g/km)',
        orientation='h',
        color='Type',
        color_discrete_map={
            'Target': '#00e5ff',       # Cyan for user specs
            'Alternative': '#00c04b'  # Green for recommendations
        },
        text='CO2 Emissions (g/km)',
        labels={'Label': 'Vehicle Model'},
        title="CO₂ Emissions Comparison Chart"
    )
    
    # Style Chart
    fig_bar.update_traces(
        texttemplate='%{text:.0f} g/km', 
        textposition='outside', 
        cliponaxis=False,
        marker_line_color='rgba(0,0,0,0)',
        width=0.5
    )
    
    fig_bar.update_layout(
        paper_bgcolor="rgba(30, 30, 47, 0.4)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit", color="#a0a0b0"),
        title_font=dict(color="#00e5ff"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=min(400, 100 + len(comp_df) * 45),
        showlegend=False,
        xaxis_title="CO₂ Emissions (g/km)"
    )
    fig_bar.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", range=[0, max(comp_df['CO2 Emissions (g/km)']) * 1.25])
    fig_bar.update_yaxes(showgrid=False, autorange="reversed")
    
    st.plotly_chart(fig_bar, use_container_width=True)
