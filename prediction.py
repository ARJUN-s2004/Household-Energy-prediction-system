import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and data
model = joblib.load("power_consumption_model.pkl")
df = pd.read_csv("code/household_power_consumption.csv")
df.dropna(inplace=True)

# Prepare features & target
features = ['Global_reactive_power', 'Voltage', 'Global_intensity',
            'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
target = 'Global_active_power'
X = df[features]
y = df[target]
preds = model.predict(X)

# Set Streamlit config
st.set_page_config(page_title="Power Consumption Insights", layout="wide")
st.title("🔋 Power Consumption Prediction & Insights")
st.markdown("Use the sidebar to input values or view trends.")

# --- Sidebar Input ---
st.sidebar.header("⚙️ Predict Consumption")

voltage = st.sidebar.slider("Voltage (V)", 220.0, 255.0, 240.0)
reactive = st.sidebar.slider("Global Reactive Power (kW)", 0.0, 1.5, 0.1)
intensity = st.sidebar.slider("Global Intensity (A)", 0.0, 30.0, 5.0)
sm1 = st.sidebar.slider("Sub Metering 1 (Kitchen)", 0.0, 20.0, 1.0)
sm2 = st.sidebar.slider("Sub Metering 2 (Laundry)", 0.0, 20.0, 1.0)
sm3 = st.sidebar.slider("Sub Metering 3 (Heating/AC)", 0.0, 30.0, 5.0)

if st.sidebar.button("🔍 Predict"):
    input_data = np.array([[reactive, voltage, intensity, sm1, sm2, sm3]])
    prediction = model.predict(input_data)[0]
    st.sidebar.success(f"Predicted Global Active Power: {prediction:.3f} kW")

    st.sidebar.subheader("💡 Recommendation")
    if prediction > 5:
        st.sidebar.warning("⚠️ High usage detected. Reduce appliance runtime.")
    elif sm3 > 10:
        st.sidebar.info("Consider reducing heating/AC usage.")
    else:
        st.sidebar.success("Usage is optimal. Keep it up!")

# --- Main Area ---
tab1, tab2 = st.tabs(["📊 Insights", "📈 Trends & Performance"])

with tab1:
    st.subheader("Feature Impact on Energy Usage")
    corr = df[features + [target]].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr[[target]].drop(target).sort_values(by=target, ascending=False),
                annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.markdown("**Key Observations:**")
    st.markdown("- `Global_intensity` and `Voltage` have strong correlation with active power.")
    st.markdown("- Meter readings (`Sub_metering_1/2/3`) offer appliance-specific influence.")
    st.markdown("- Reducing `Global_reactive_power` may improve energy efficiency.")

with tab2:
    st.subheader("Actual vs. Predicted Power Consumption")
    sample_df = df.copy()
    sample_df["Predicted"] = preds
    sample_df["DateTime"] = pd.to_datetime(sample_df["Date"] + ' ' + sample_df["Time"],
                                           format="%d/%m/%Y %H:%M:%S", errors='coerce')

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sample_df_sorted = sample_df.sort_values(by="DateTime").dropna(subset=["DateTime"])
    ax2.plot(sample_df_sorted["DateTime"][:500], sample_df_sorted[target][:500], label="Actual", alpha=0.7)
    ax2.plot(sample_df_sorted["DateTime"][:500], sample_df_sorted["Predicted"][:500], label="Predicted", alpha=0.7)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Power (kW)")
    ax2.legend()
    st.pyplot(fig2)

    st.markdown("The model generally tracks real usage well. Outliers may signal unusual appliance behavior.")

