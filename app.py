import streamlit as st
import pandas as pd
import joblib
import sklearn

st.markdown("<h1 style='text-align: center;'>🚗 Car Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #555; margin-top: -10px;'>What’s Your Car Worth? Find Out!</h4>", unsafe_allow_html=True)


required_version = "1.6.1"
current_version = sklearn.__version__
if current_version != required_version:
    st.warning(f"⚠️ Model trained on scikit-learn {required_version}, but you're using {current_version}. Compatibility issues may occur.")

try:
    model = joblib.load('car_price_model.pkl')
    car_data = joblib.load('car_data.pkl')
except Exception as e:
    st.error(f"❌ Failed to load model or data: {e}")
    st.stop()

# Select brand first (outside form for instant update)
company = st.selectbox("🔹 Select Car Brand", sorted(car_data['company']))

# Update models dropdown immediately after brand selection
if company:
    # Make sure model names start with brand + space for exact match
    car_models = sorted([name for name in car_data['name'] if name.startswith(company + " ")])
    if car_models:
        car_model = st.selectbox("🔸 Select Car Model", car_models)
    else:
        st.warning("⚠️ No models found for this brand.")
        car_model = None
else:
    st.info("ℹ️ Please select a car brand to see available models.")
    car_model = None

# Now the rest of inputs and submit button inside a form
with st.form(key='car_form'):
    year = st.number_input("📅 Year of Purchase", min_value=1990, max_value=2025, value=2015, step=1)
    fuel_type = st.selectbox("⛽ Fuel Type", sorted(car_data['fuel_type']))
    kms_driven = st.number_input("🛣️ Kilometers Driven", min_value=0, max_value=1000000, value=50000, step=1)
    submit_button = st.form_submit_button(label="🔍 Predict Price")

if submit_button:
    if not company:
        st.error("❗ Please select a valid car brand.")
    elif not car_model:
        st.error("❗ Please select a valid car model.")
    else:
        input_df = pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]],
                                columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
        try:
            prediction = model.predict(input_df)[0]
            st.success(f"💰 Estimated Car Price: ₹ {round(prediction, 2):,}")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
