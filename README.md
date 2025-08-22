# CarboSense: GenAI-Powered Carbon Footprint Analyzer 
CarboSense is a web application that helps users understand their environmental impact by calculating their carbon footprint based on their lifestyle choices. It uses a machine learning model to predict the total footprint and leverages a Generative AI (GenAI) model to provide personalized, actionable suggestions for reduction.

## Features
* Comprehensive Data Input: Users can input data across various categories including household energy, transportation, diet, and waste.

* ML-Powered Prediction: Utilizes a trained XGBoost model to accurately predict the user's monthly carbon footprint in kgCO2e.

* Detailed Breakdown: Visualizes the carbon footprint breakdown by category, helping users identify their biggest impact areas.

* AI-Powered Suggestions: Connects to the Gemini API to generate personalized, friendly, and practical tips for reducing one's carbon footprint.

* Responsive Frontend: A beautiful and intuitive user interface built with HTML and Tailwind CSS that works on all devices.

* FastAPI Backend: A robust and efficient backend server that handles data processing, model inference, and API communication.

## Tech Stack_
* Frontend:

HTML5

Tailwind CSS

* Backend:

Python 3.11.24

FastAPI

Uvicorn (ASGI Server)

* Machine Learning:

Scikit-learn

XGBoost

Pandas

Joblib (for model serialization)

* Generative AI:

Google Gemini API (You can use any GenAI model)

