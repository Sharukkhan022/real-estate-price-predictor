from pathlib import Path

import streamlit as st
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'models' / 'linear_regression_model.pkl'
DATA_PATH = BASE_DIR / 'data' / 'redfin_housing.csv'

# Set page title and configuration
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

st.markdown(
    """
    <style>
        :root {
            --bg: #0f172a;
            --panel: rgba(15, 23, 42, 0.78);
            --panel-2: rgba(30, 41, 59, 0.9);
            --primary: #7c3aed;
            --primary-2: #22c55e;
            --accent: #38bdf8;
            --text: #e2e8f0;
            --muted: #cbd5e1;
            --success: #16a34a;
        }

        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #111827 35%, #1e293b 100%);
            color: var(--text);
        }

        .main .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3 {
            color: #f8fafc !important;
        }

        .stTitle {
            background: linear-gradient(90deg, #a78bfa, #67e8f9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            letter-spacing: 0.02em;
        }

        .stMarkdown p, .stMarkdown li {
            color: var(--muted) !important;
        }

        .stButton > button {
            background: linear-gradient(135deg, var(--primary), var(--accent));
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.7rem 1.4rem;
            font-weight: 700;
            box-shadow: 0 10px 24px rgba(124, 58, 237, 0.35);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 14px 30px rgba(56, 189, 248, 0.35);
        }

        .stSelectbox label, .stNumberInput label, .stSlider label {
            color: #e2e8f0 !important;
            font-weight: 600;
        }

        .stDataFrame, .stTable {
            background: rgba(15, 23, 42, 0.7);
            border-radius: 16px;
        }

        .stSuccess {
            background: linear-gradient(135deg, rgba(22, 163, 74, 0.18), rgba(34, 197, 94, 0.25));
            border: 1px solid rgba(34, 197, 94, 0.25);
            border-radius: 16px;
            color: #dcfce7;
            padding: 1rem 1.2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏠 House Price Prediction App")
st.write("Enter the property details below to predict the estimated market price.")

# Load the trained model
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run 'python src/train_model.py' first."
        )
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_city_price_map():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}.")

    df = pd.read_csv(DATA_PATH)
    city_map = df.groupby('City')['Price (USD)'].mean().to_dict()
    return city_map

model = load_model()
city_price_map = load_city_price_map()
overall_mean_price = pd.read_csv(DATA_PATH)['Price (USD)'].mean()

# Create input form for user
st.header("Property Details")

col1, col2 = st.columns(2)

with col1:
    beds = st.number_input("Number of Bedrooms", min_value=1.0, max_value=10.0, value=3.0, step=1.0)
    baths = st.number_input("Number of Bathrooms", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
    area = st.number_input("Area (SQFT)", min_value=200, max_value=20000, value=1500, step=50)

with col2:
    sale_year = st.selectbox("Sale Year", [2023, 2024, 2025, 2026], index=0)
    sale_month = st.slider("Sale Month", min_value=1, max_value=12, value=6)
    cities = sorted(city_price_map.keys())
    selected_city = st.selectbox("City", cities)

# Prediction trigger
if st.button("Predict House Price 🚀", use_container_width=True):
    # Convert input city to numeric value using your target encoding
    city_encoded = city_price_map.get(selected_city, overall_mean_price)

    # Format input into DataFrame with exact feature names used during model training
    input_data = pd.DataFrame([{
        'Beds': beds,
        'Baths': baths,
        'Area (SQFT)': area,
        'Sale_Year': sale_year,
        'Sale_Month': sale_month,
        'City_Avg_Price': city_encoded
    }])

    # Generate prediction
    prediction = model.predict(input_data)[0]

    st.success(f"### Estimated Price: **${prediction:,.2f}**")