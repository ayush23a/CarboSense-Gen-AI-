# main.py
# To run this: 
# 1. pip install fastapi uvicorn python-multipart pandas joblib scikit-learn xgboost
# 2. You will also need a trained model file named 'xgboost_model.joblib' and 'model_columns.joblib'. 
#    You can create this by adding the necessary code to the end of your Jupyter notebook and running it.
# 3. Run the server with: uvicorn main:app --reload

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import json
import requests 
import os
from dotenv import load_dotenv

load_dotenv() 

# --- Pydantic Model for Input Validation ---

class UserInput(BaseModel):
    electricity_kwh_month: int
    natural_gas_kwh_month: int
    car_km_month: int
    car_fuel_type: str
    car_avg_mileage_km_per_liter: float
    bus_km_month: int
    train_km_month: int
    flights_short_haul_per_year: int
    flights_long_haul_per_year: int
    red_meat_meals_per_week: int
    chicken_meals_per_week: int
    veg_meals_per_week: int
    waste_kg_month: float
    is_recycler: int
    ac_hours_day_summer: float
    num_people_household: int


app = FastAPI(
    title="CarboSense API",
    description="API for calculating carbon footprint and getting AI suggestions.",
    version="1.0.0"
)

# --- CORS Middleware ---

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST
    allow_headers=["*"],  
)

# --- Load Trained Model and Columns ---
try:
    model = joblib.load('xgboost_model.joblib')
    model_columns = joblib.load('model_columns.joblib') 
except FileNotFoundError:
    model = None
    model_columns = None
    print("WARNING: 'xgboost_model.joblib' or 'model_columns.joblib' not found.")
    print("The /analyze endpoint will not work without a trained model.")


#Emission Factors 
EMISSION_FACTORS = {
    'electricity_kwh': 0.8, 'natural_gas_kwh': 0.19, 'petrol_liter': 2.3,
    'diesel_liter': 2.65, 'bus_km': 0.1, 'train_km': 0.096,
    'flight_short_haul_km': 0.15, 'flight_long_haul_km': 0.1, 'red_meat_meal': 10.0,
    'chicken_meal': 1.6, 'veg_meal': 0.3, 'waste_kg': 2.0,
}

# Helper Functions 
def calculate_footprint_components(data: dict, emission_factors: dict) -> dict:
    """Calculates the breakdown of carbon footprint by category."""
    
    row = pd.Series(data)
    
    
    cf_electricity = row['electricity_kwh_month'] * emission_factors['electricity_kwh']
    cf_gas = row['natural_gas_kwh_month'] * emission_factors['natural_gas_kwh']
    # Calculate AC kWh from hours, assuming 1.5 kW consumption
    ac_kwh_month = row['ac_hours_day_summer'] * 1.5 * 30 * (4/12)
    cf_ac = ac_kwh_month * emission_factors['electricity_kwh']

    # Transportation
    cf_car = 0
    if row['car_fuel_type'] == 'petrol' and row['car_avg_mileage_km_per_liter'] > 0:
        liters = row['car_km_month'] / row['car_avg_mileage_km_per_liter']
        cf_car = liters * emission_factors['petrol_liter']
    elif row['car_fuel_type'] == 'diesel' and row['car_avg_mileage_km_per_liter'] > 0:
        liters = row['car_km_month'] / row['car_avg_mileage_km_per_liter']
        cf_car = liters * emission_factors['diesel_liter']
        
    cf_bus = row['bus_km_month'] * emission_factors['bus_km']
    cf_train = row['train_km_month'] * emission_factors['train_km']
    
    # Flights
    cf_flights_short = (row['flights_short_haul_per_year'] * 1000 * emission_factors['flight_short_haul_km']) / 12
    cf_flights_long = (row['flights_long_haul_per_year'] * 5000 * emission_factors['flight_long_haul_km']) / 12
    
    # Diet
    monthly_factor = 52 / 12
    cf_diet_red_meat = (row['red_meat_meals_per_week'] * emission_factors['red_meat_meal']) * monthly_factor
    cf_diet_chicken = (row['chicken_meals_per_week'] * emission_factors['chicken_meal']) * monthly_factor
    cf_diet_veg = (row['veg_meals_per_week'] * emission_factors['veg_meal']) * monthly_factor
    
    # Waste
    cf_waste = row['waste_kg_month'] * emission_factors['waste_kg']
    if row['is_recycler'] == 1:
        cf_waste *= 0.2

    breakdown = {
        'Electricity_and_AC': cf_electricity + cf_ac,
        'Natural_Gas': cf_gas,
        'Car_Travel': cf_car,
        'Public_Transport': cf_bus + cf_train,
        'Flights': cf_flights_short + cf_flights_long,
        'Diet': cf_diet_red_meat + cf_diet_chicken + cf_diet_veg,
        'Waste': cf_waste,
    }
    
    sorted_breakdown = dict(sorted(breakdown.items(), key=lambda item: item[1], reverse=True))
    return sorted_breakdown

async def generate_ai_suggestions(cf_breakdown_data: dict) -> str:
    """Generates personalized suggestions using the Gemini API."""
    prompt = (
        "Given the following monthly carbon footprint breakdown for a person (in kgCO2e), "
        "please provide 3-5 actionable and friendly suggestions to help them reduce their environmental impact. "
        "Focus on the areas with the highest contribution. Ensure the tone is encouraging and practical, avoiding jargon.\n\n"
        "Carbon Footprint Breakdown:\n"
    )
    
    for category, value in cf_breakdown_data.items():
        prompt += f"- {category.replace('_', ' ')}: {value:.2f} kgCO2e/month\n"
    
    prompt += "\nSuggestions:"

    apiKey = os.getenv("GOOGLE_API_KEY") 
    if not apiKey:
        return "API key not found. Please set the GOOGLE_API_KEY environment variable."
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={apiKey}"
    
    payload = { "contents": [{ "parts": [{ "text": prompt }] }] }

    try:
        response = requests.post(apiUrl, headers={"Content-Type": "application/json"}, json=payload)
        response.raise_for_status() 
        result = response.json()

        if (result.get('candidates') and len(result['candidates']) > 0 and 
            result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts') and
            len(result['candidates'][0]['content']['parts']) > 0):
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            error_message = result.get('error', {}).get('message', 'Unknown error from API.')
            return f"Could not generate AI suggestions. API returned: {error_message}"

    except Exception as e:
        print(f"An error occurred while generating AI suggestions: {e}")
        return f"An error occurred while generating AI suggestions: {e}"


# API Endpoints 
@app.get("/")
def read_root():
    return {"message": "Welcome to the CarboSense API. Go to /docs for documentation."}

@app.post("/analyze")
async def analyze_footprint(user_input: UserInput):
    """
    Analyzes user input to predict carbon footprint, provide a breakdown,
    and generate AI-powered suggestions.
    """
    if not model or not model_columns:
        raise HTTPException(
            status_code=500, 
            detail="Machine learning model is not loaded. Please check server configuration."
        )
        
    input_df = pd.DataFrame([user_input.dict()])

    # Feature Engineering 
    
    input_df['ac_kwh_month'] = input_df['ac_hours_day_summer'] * 1.5 * 30 * (4/12)
    
    input_df = pd.get_dummies(input_df, columns=['car_fuel_type'], drop_first=False)
    
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    
    input_df = input_df[model_columns]

    # Prediction and Analysis 
    predicted_cf = model.predict(input_df)[0]
    breakdown = calculate_footprint_components(user_input.dict(), EMISSION_FACTORS)
    suggestions = await generate_ai_suggestions(breakdown)

    return {
        "total_footprint": float(predicted_cf),
        "breakdown": breakdown,
        "suggestions": suggestions
    }
