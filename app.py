import streamlit as st
import pickle
import pandas as pd

# ------------------ Page Configuration ------------------
st.set_page_config(
    page_title="Sales Forecasting System",
    page_icon="📈",
    layout="centered"
)

# ------------------ Custom Styling ------------------
st.markdown("""
    <style>
    .main {
        background-color: #f7f9fc;
    }
    .title {
        font-size: 36px;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 18px;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    .card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }
    .result {
        font-size: 28px;
        font-weight: 600;
        color: #1e8449;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ Load Model ------------------
model = pickle.load(open("sales_model.pkl", "rb"))

# ------------------ Title ------------------
st.markdown('<div class="title">📈 Sales Forecasting System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict future sales using historical data</div>', unsafe_allow_html=True)

# ------------------ Input Section ------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🔍 Select Forecast Date")
st.write("Choose a future date to estimate expected sales.")
forecast_date = st.date_input("Forecast Date")
st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Prediction ------------------
if st.button("🚀 Predict Sales"):
    future_df = pd.DataFrame({
        "ds": [pd.to_datetime(forecast_date)]
    })

    forecast = model.predict(future_df)
    predicted_sales = forecast["yhat"].iloc[0]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="result">Estimated Sales<br>₹ {predicted_sales:,.2f}</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Footer ------------------
st.markdown(
    "<hr><center><small>Sales Forecasting Project | Streamlit Deployment</small></center>",
    unsafe_allow_html=True
)
