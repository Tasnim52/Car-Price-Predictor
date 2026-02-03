import streamlit as st
import pandas as pd
import pickle

# 1. Page Configuration
st.set_page_config(
    page_title="AutoPredict AI",
    page_icon="🏎️",
    layout="wide"
)

# 2. Custom Designer CSS
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #e63946;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #d62828;
        color: white;
    }
    .prediction-card {
        padding: 30px;
        border-radius: 15px;
        background-color: #ffffff;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 5px solid #e63946;
    }
    .price-text {
        color: #1d3557;
        font-size: 45px;
        font-weight: 800;
    }
    </style>
    """, unsafe_allow_html=True)


# 3. Load Models and UI Options
@st.cache_resource
def load_assets():
    try:
        with open('car_price_models.pkl', 'rb') as f:
            models = pickle.load(f)
        with open('ui_data.pkl', 'rb') as f:
            ui = pickle.load(f)
        return models, ui
    except FileNotFoundError:
        st.error("Model files not found! Please run train_model.py first.")
        return None, None


models, ui = load_assets()

# 4. Image Mapping for Designer Look
brand_images = {
    "BMW": "https://images.unsplash.com/photo-1555215695-3004980ad54e?q=80&w=800",
    "Tesla": "https://images.unsplash.com/photo-1560958089-b8a1929cea89?q=80&w=800",
    "Mercedes": "https://images.unsplash.com/photo-1618843479313-40f8afb4b4d8?q=80&w=800",
    "Audi": "https://images.unsplash.com/photo-1606152421802-db97b9c7a11b?q=80&w=800",
    "Toyota": "https://images.unsplash.com/photo-1621007947382-bb3c3994e3fb?q=80&w=800",
    "Hyundai": "https://images.unsplash.com/photo-1503376780353-7e6692767b70?q=80&w=800",
    "Volkswagen": "https://images.unsplash.com/photo-1541899481282-d53bffe3c35d?q=80&w=800"
}
default_img = "https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?q=80&w=800"

# 5. UI Layout
if models and ui:
    st.title("🚗 Premium Car Price Predictor")
    st.write("Enter details below to get an AI-powered valuation.")
    st.markdown("---")

    col1, col2 = st.columns([1, 1.5], gap="large")

    with col1:
        st.subheader("📋 Car Specifications")

        brand = st.selectbox("Select Brand", ui['brands'])
        # Filter models based on brand could be added here, but using full list for now
        model_name = st.selectbox("Select Model", ui['models'])

        year = st.slider("Registration Year", 2015, 2026, 2022)
        engine = st.number_input("Engine Capacity (Liters)", 1.0, 6.5, 2.0, step=0.1)
        mileage = st.number_input("Total Kilometers Driven", 0, 200000, 25000, step=500)

        f1, f2 = st.columns(2)
        with f1:
            fuel = st.selectbox("Fuel Type", ui['fuel_types'])
        with f2:
            trans = st.selectbox("Transmission", ui['transmissions'])

        cond = st.radio("Vehicle Condition", ui['conditions'], horizontal=True)

        st.markdown("---")
        selected_algo = st.selectbox("🤖 Machine Learning Model", list(models.keys()))

    with col2:
        st.subheader("📷 Vehicle Preview")
        # Dynamic Image based on Brand
        st.image(brand_images.get(brand, default_img), use_container_width=True,
                 caption=f"Typical {brand} configuration")

        st.write(" ")  # Spacing

        if st.button("CALCULATE ESTIMATED PRICE"):
            # Create input dataframe for prediction
            input_data = pd.DataFrame([{
                'Brand': brand,
                'Year': year,
                'Engine Size': engine,
                'Fuel Type': fuel,
                'Transmission': trans,
                'Mileage': mileage,
                'Condition': cond,
                'Model': model_name
            }])

            # Predict
            try:
                prediction = models[selected_algo].predict(input_data)[0]

                # Display Result in a nice card
                st.markdown(f"""
                    <div class="prediction-card">
                        <h3>Valuation Result</h3>
                        <p class="price-text">₹ {prediction:,.2f}</p>
                        <p style="color: grey;">Model Used: {selected_algo}</p>
                        <p><i>The estimated price is based on historical market data and vehicle condition.</i></p>
                    </div>
                """, unsafe_allow_html=True)
                st.balloons()
            except Exception as e:
                st.error(f"Prediction Error: {e}")

    st.markdown("---")
    st.caption("© 2026 CarPrice AI Intern Project | Built with Streamlit")