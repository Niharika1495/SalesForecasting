import streamlit as st
import pickle
import pandas as pd

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Sales Forecasting System",
    page_icon="📈",
    layout="centered"
)

# ------------------ Custom CSS ------------------
st.markdown("""
<style>
.main { background-color: #f7f9fc; }
.title {
    font-size: 36px;
    font-weight: 700;
    color: #2c3e50;
    text-align: center;
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
    font-size: 24px;
    font-weight: 600;
    text-align: center;
}
.profit { color: #1e8449; }
.loss { color: #c0392b; }
</style>
""", unsafe_allow_html=True)

# ------------------ Load Model ------------------
model = pickle.load(open("sales_model.pkl", "rb"))

# ------------------ Load Training Data ------------------
df = pd.read_csv("train.csv")

# ---- Adjust column names if needed ----
# If your dataset already uses ds and y, this will do nothing
if "date" in df.columns:
    df.rename(columns={"date": "ds"}, inplace=True)
if "sales" in df.columns:
    df.rename(columns={"sales": "y"}, inplace=True)

df["ds"] = pd.to_datetime(df["ds"])

# ------------------ Title ------------------
st.markdown('<div class="title">📈 Sales Forecasting System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Compare present sales with future predicted sales</div>',
    unsafe_allow_html=True
)

# ------------------ Present Sales ------------------
latest_date = df["ds"].max()
present_sales = df.loc[df["ds"] == latest_date, "y"].values[0]

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📌 Present Sales")
st.write(f"Latest Available Date: **{latest_date.date()}**")
st.write(f"Sales Value: **₹ {present_sales:,.2f}**")
st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Future Prediction ------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🔮 Future Sales Prediction")
forecast_date = st.date_input("Select future date")

if st.button("🚀 Predict & Compare"):
    future_df = pd.DataFrame({
        "ds": [pd.to_datetime(forecast_date)]
    })

    forecast = model.predict(future_df)
    future_sales = forecast["yhat"].iloc[0]

    difference = future_sales - present_sales

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="result">Future Sales<br>₹ {future_sales:,.2f}</div>',
        unsafe_allow_html=True
    )

    if difference > 0:
        st.markdown(
            f'<div class="result profit">📈 PROFIT of ₹ {difference:,.2f}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result loss">📉 LOSS of ₹ {abs(difference):,.2f}</div>',
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Footer ------------------
st.markdown(
    "<hr><center><small>Sales Forecasting Project | Streamlit Deployment</small></center>",
    unsafe_allow_html=True
)
