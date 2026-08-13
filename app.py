import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration
st.set_page_config(page_title="CO2 Emission Predictor", page_icon="🚗", layout="centered")

st.title("Vehicle CO2 Emission Predictor 🚗💨")
st.write("Enter the specifications of a vehicle below to estimate its CO2 emissions based on our Random Forest model.")

# 2. Load the Model
# We use @st.cache_resource so the app only loads the model once, making it faster.
@st.cache_resource
def load_model():
    # Tell Streamlit to look inside the 'model' folder
    return joblib.load('model/co2_rf_model.pkl')

model = load_model()

# 3. Define Category Lists for Dropdowns
makes = [
    'Acura', 'Alfa Romeo', 'Aston Martin', 'Audi', 'BMW', 'Bentley', 'Bugatti', 
    'Buick', 'Cadillac', 'Chevrolet', 'Chrysler', 'Dodge', 'FIAT', 'Ferrari', 
    'Ford', 'GMC', 'Genesis', 'Honda', 'Hyundai', 'Infiniti', 'Jaguar', 'Jeep', 
    'Kia', 'Lamborghini', 'Land Rover', 'Lexus', 'Lincoln', 'MINI', 'Maserati', 
    'Mazda', 'Mercedes-Benz', 'Mitsubishi', 'Nissan', 'Porsche', 'Ram', 
    'Rolls-Royce', 'Scion', 'Subaru', 'Toyota', 'Volkswagen', 'Volvo', 'smart'
]

vehicle_classes = [
    'Compact', 'Full-size', 'Mid-size', 'Minicompact', 'Minivan', 
    'Pickup truck: Small', 'Pickup truck: Standard', 'Special purpose vehicle', 
    'Sport utility vehicle: Small', 'Sport utility vehicle: Standard', 
    'Station wagon: Mid-size', 'Station wagon: Small', 'Subcompact', 
    'Two-seater', 'Van: Passenger'
]

# 4. Create the User Input Form
st.markdown("### Vehicle Specifications")

# Using columns to make the layout look clean
col1, col2 = st.columns(2)

with col1:
    # Dropdowns for categorical features 
    make = st.selectbox("Make", options=makes)
    vehicle_class = st.selectbox("Vehicle Class", options=vehicle_classes)
    
    # Number inputs for numerical features
    engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
    model_year = st.number_input("Model Year", min_value=1990, max_value=2025, value=2015, step=1)

with col2:
    # Dropdowns for categorical features with specific known categories
    # The format_func makes the UI friendly while sending the correct letter (X, Z, D, E) to the model
    fuel_type = st.selectbox(
        "Fuel Type", 
        options = ["X", "Z", "D", "E"],
        format_func = lambda x: {
            "X": "Regular Gasoline (X)", 
            "Z": "Premium Gasoline (Z)", 
            "D": "Diesel (D)", 
            "E": "Ethanol (E)"
        }.get(x)
    )
    
    transmission = st.selectbox("Transmission Type", options = ["A", "AM", "AS", "AV", "M"], index=2)
    gears = st.number_input("Gears", min_value = 1, max_value = 10, value = 5, step = 1)

# 5. The Prediction Button
st.markdown("---")
if st.button("Predict CO2 Emissions", type = "primary"):
    
    # Construct a DataFrame exactly how the notebook model expects it.
    # CRITICAL: Column names and order must match the training data perfectly.
    input_data = pd.DataFrame({
        'Model year': [model_year],
        'Make': [make],
        'Vehicle class': [vehicle_class],
        'Engine size (L)': [engine_size],
        'Fuel type': [fuel_type],
        'Transmission type': [transmission],
        'Gears': [gears]
    })
    
    # Run the prediction through the loaded pipeline
    prediction = model.predict(input_data)
    
    # Display the result
    st.success(f"### Estimated CO2 Emission: {prediction[0]:.2f} g/km")