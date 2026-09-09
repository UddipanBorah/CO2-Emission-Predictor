import os
import joblib
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def fit_and_save_recommender(model_path='model/co2_production_model.pkl', 
                              data_path='datasets/feature_engineered_data/feature_engineered_cars.csv',
                              recommender_output_path='model/co2_knn_recommender.pkl'):
    """
    Fits NearestNeighbors on the preprocessed feature matrix of the dataset and saves it.
    Excludes Fuel type == 'N' from the recommendation pool.
    """
    if not os.path.exists(model_path):
        # Fallback for parent directories if running from subfolders
        model_path = os.path.join('..', model_path)
        data_path = os.path.join('..', data_path)
        recommender_output_path = os.path.join('..', recommender_output_path)
        
    # Load production pipeline
    pipeline = joblib.load(model_path)
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Load feature engineered data
    df = pd.read_csv(data_path)
    
    # Load cleaned data to retrieve the Model column (which was dropped for training but is required for recommendations)
    cleaned_data_path = data_path.replace('feature_engineered_data/feature_engineered_cars.csv', 'cleaned_data/cleaned_cars_info.csv')
    if not os.path.exists(cleaned_data_path) and os.path.exists(os.path.join('..', cleaned_data_path)):
        cleaned_data_path = os.path.join('..', cleaned_data_path)
    df_cleaned = pd.read_csv(cleaned_data_path)
    df['Model'] = df_cleaned['Model']
    
    # Exclude Natural Gas (N) fuel type rows
    df_filtered = df[df['Fuel type'] != 'N'].copy().reset_index(drop=True)
    
    # Define features and target
    feature_cols = ['Model year', 'Make', 'Vehicle class', 'Engine size (L)', 'Fuel type', 'Transmission type', 'Gears']
    X = df_filtered[feature_cols]
    
    # Transform feature matrix using the exact fitted preprocessor from the production model
    X_encoded = preprocessor.transform(X)
    
    # Fit KNN with cosine similarity
    knn = NearestNeighbors(n_neighbors=100, metric='cosine')
    knn.fit(X_encoded)
    
    # Save both the fitted KNN model and the filtered original dataframe for quick lookup
    payload = {
        'knn': knn,
        'reference_df': df_filtered,
        'feature_cols': feature_cols
    }
    
    os.makedirs(os.path.dirname(recommender_output_path), exist_ok=True)
    joblib.dump(payload, recommender_output_path)
    print(f"Recommender system fitted and saved to {recommender_output_path}")

def recommend_lower_emission(user_specs_df, predicted_co2, k=5, recommender_path='model/co2_knn_recommender.pkl', model_path='model/co2_production_model.pkl'):
    """
    Given a user's vehicle specs and predicted CO2 emission, returns up to k similar vehicles
    from the reference pool that have lower actual emissions.
    """
    if not os.path.exists(recommender_path):
        recommender_path = os.path.join('..', recommender_path)
        model_path = os.path.join('..', model_path)
        
    payload = joblib.load(recommender_path)
    knn = payload['knn']
    reference_df = payload['reference_df']
    feature_cols = payload['feature_cols']
    
    pipeline = joblib.load(model_path)
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Preprocess user input
    user_encoded = preprocessor.transform(user_specs_df[feature_cols])
    
    # Start by querying a larger number of neighbors to ensure we find enough with lower emissions
    distances, indices = knn.kneighbors(user_encoded, n_neighbors=min(150, len(reference_df)))
    
    distances = distances[0]
    indices = indices[0]
    
    # Retrieve matching rows
    neighbors_df = reference_df.iloc[indices].copy()
    neighbors_df['Similarity'] = 1 - distances # Cosine similarity = 1 - Cosine distance
    
    # Filter to only include vehicles with actual emissions lower than predicted
    # Also exclude the exact same vehicle make/model if it has identical specs (optional, but keep it for comparison)
    alternatives = neighbors_df[neighbors_df['CO2 emissions (g/km)'] < predicted_co2]
    
    # If we don't have enough, we can return what we have or search more.
    # We sort by similarity descending, then by CO2 emissions ascending
    alternatives = alternatives.sort_values(by=['Similarity', 'CO2 emissions (g/km)'], ascending=[False, True])
    
    # Return top k
    cols_to_return = ['Make', 'Model', 'Vehicle class', 'Engine size (L)', 'Fuel type', 'Transmission type', 'Gears', 'CO2 emissions (g/km)', 'Similarity']
    return alternatives[cols_to_return].head(k)

if __name__ == "__main__":
    # If run directly, fit the recommender
    fit_and_save_recommender()
