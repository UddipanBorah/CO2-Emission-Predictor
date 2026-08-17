import os
import joblib
import pandas as pd
import streamlit as st
from recommender import recommend_lower_emission

@st.cache_resource
def load_model(model_path='model/co2_production_model.pkl'):
    """
    Loads and caches the production pipeline.
    """
    if not os.path.exists(model_path):
        model_path = os.path.join('..', model_path)
    return joblib.load(model_path)

@st.cache_resource
def load_recommender(recommender_path='model/co2_knn_recommender.pkl'):
    """
    Loads and caches the fitted recommender structures.
    """
    if not os.path.exists(recommender_path):
        recommender_path = os.path.join('..', recommender_path)
    return joblib.load(recommender_path)

def predict_co2(input_dict):
    """
    Takes a dictionary of features, runs it through the pipeline,
    and returns the predicted CO2 emissions (g/km) as a float.
    """
    # Convert input dict to DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # Ensure correct column order
    feature_cols = ['Model year', 'Make', 'Vehicle class', 'Engine size (L)', 'Fuel type', 'Transmission type', 'Gears']
    input_df = input_df[feature_cols]
    
    model = load_model()
    prediction = model.predict(input_df)[0]
    return float(prediction)

def get_recommendations(input_dict, predicted_co2, k=5):
    """
    Takes the user inputs and prediction, and returns a DataFrame of eco-friendly recommendations.
    """
    input_df = pd.DataFrame([input_dict])
    
    # Ensure correct column order
    feature_cols = ['Model year', 'Make', 'Vehicle class', 'Engine size (L)', 'Fuel type', 'Transmission type', 'Gears']
    input_df = input_df[feature_cols]
    
    # Call the recommender core function
    return recommend_lower_emission(input_df, predicted_co2, k=k)

@st.cache_data
def get_categorical_options():
    """
    Returns unique sorted categorical options from the dataset for UI dropdowns.
    """
    payload = load_recommender()
    ref_df = payload['reference_df']
    return {
        'Make': sorted(ref_df['Make'].unique().tolist()),
        'Vehicle class': sorted(ref_df['Vehicle class'].unique().tolist()),
        'Fuel type': sorted(ref_df['Fuel type'].unique().tolist()),
        'Transmission type': sorted(ref_df['Transmission type'].unique().tolist())
    }

