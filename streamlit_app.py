import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load trained model and metadata
model, imputer, feature_cols = pickle.load(open("monaco_model.pkl", "rb"))

# --- PAGE CONFIG ---
st.set_page_config(page_title="F1 Monaco GP 2025 Predictor", page_icon="🏎️", layout="wide")

# --- HEADER ---
st.markdown("<h1 style='text-align: center; color: red;'>🏁 F1 Monaco GP 2025 Race Pace Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter qualifying times and conditions to simulate the 2025 Monaco GP</p>", unsafe_allow_html=True)
st.markdown("---")

# --- DRIVER / TEAM DATA ---
drivers_2025 = {
    "McLaren": ["Lando Norris", "Oscar Piastri"],
    "Ferrari": ["Charles Leclerc", "Lewis Hamilton"],
    "Red Bull Racing": ["Max Verstappen", "Yuki Tsunoda"],
    "Mercedes": ["George Russell", "Kimi Antonelli"],
    "Aston Martin": ["Fernando Alonso", "Lance Stroll"],
    "Racing Bulls": ["Liam Lawson", "Isack Hadjar"],
    "Haas F1 Team": ["Esteban Ocon", "Oliver Bearman"],
    "Kick Sauber": ["Nico Hülkenberg", "Gabriel Bortoleto"],
    "Williams": ["Alexander Albon", "Carlos Sainz"],
    "Alpine": ["Pierre Gasly", "Franco Colapinto"]
}

# Load historical Monaco features
hist = pd.read_csv("driver_year.csv")

# Monaco experience map
monaco_exp = {
    "Lando Norris": 3, "Oscar Piastri": 1,
    "Charles Leclerc": 5, "Lewis Hamilton": 15,
    "Max Verstappen": 8, "Yuki Tsunoda": 3,
    "George Russell": 4, "Kimi Antonelli": 0,
    "Fernando Alonso": 20, "Lance Stroll": 6,
    "Liam Lawson": 0, "Isack Hadjar": 0,
    "Esteban Ocon": 6, "Oliver Bearman": 0,
    "Nico Hülkenberg": 6, "Gabriel Bortoleto": 0,
    "Alexander Albon": 4, "Carlos Sainz": 7,
    "Pierre Gasly": 6, "Franco Colapinto": 0
}

# --- QUALIFYING INPUTS ---
st.subheader("📋 Enter Qualifying Times (seconds)")
st.caption("Enter 0 if the driver did not participate or failed to set a lap (will be ignored in ranking).")

input_rows = []
cols = st.columns(2)
team_list = list(drivers_2025.items())

for i, (team, drivers) in enumerate(team_list):
    with cols[i % 2]:
        st.markdown(f"### 🏎️ {team}")
        for d in drivers:
            qt = st.number_input(f"{d} Qualifying Time", min_value=0.0, max_value=200.0, step=0.001, value=0.0)
            input_rows.append({"Driver": d, "QualifyingTime": qt, "Team": team})

df = pd.DataFrame(input_rows)

# --- MERGE HISTORICAL FEATURES ---
df = df.merge(hist[["Driver", "TotalSector", "Consistency", "LapCount", "MonacoRaces"]], on="Driver", how="left")

# Replace missing values
df["MonacoRaces"] = df["Driver"].map(monaco_exp)
df["TotalSector"] = df["TotalSector"].fillna(df["TotalSector"].mean())
df["Consistency"] = df["Consistency"].fillna(df["Consistency"].mean())
df["LapCount"] = df["LapCount"].fillna(df["LapCount"].mean())
df["MonacoRaces"] = df["MonacoRaces"].fillna(0)

# --- CONDITIONS ---
st.markdown("---")
st.subheader("🌦️ Race Conditions")

col1, col2, col3 = st.columns(3)
with col1:
    temp = st.slider("Track Temperature (°C)", 15, 45, 25)
with col2:
    rain = st.slider("Rain Probability (%)", 0, 100, 0)
with col3:
    quali_weight = st.slider("Qualifying Impact (0 = minimal, 1 = high)", 0.0, 1.0, 0.6)

# --- PREDICTION ---
st.markdown("---")

if st.button("🚦 Predict Race Pace & Results"):

    # Remove zero/invalid drivers
    df_valid = df[df["QualifyingTime"] > 0].copy()

    if len(df_valid) == 0:
        st.error("Enter at least one valid qualifying time!")
        st.stop()

    # Prepare model input
    X_input = df_valid[feature_cols]
    X_input = imputer.transform(X_input)

    # Predict base race pace
    df_valid["PredictedLap"] = model.predict(X_input)

    # Adjust for qualifying influence
    q_norm = df_valid["QualifyingTime"] / df_valid["QualifyingTime"].mean()
    df_valid["PredictedLap"] = (
        (1 - quali_weight) * df_valid["PredictedLap"] +
        quali_weight * q_norm * df_valid["PredictedLap"].mean()
    )

    # Add weather penalty
    rain_factor = 1 + (rain / 100) * 0.05
    temp_factor = 1 + abs(temp - 25) * 0.005
    df_valid["PredictedLap"] = df_valid["PredictedLap"] * rain_factor * temp_factor

    # Sort for leaderboard
    df_valid = df_valid.sort_values("PredictedLap").reset_index(drop=True)
    df_valid["Position"] = df_valid.index + 1

    # --- DISPLAY RESULTS ---
    st.success("✅ Prediction Complete!")

    st.markdown("### 🏁 Predicted 2025 Monaco GP Finishing Order")
    st.dataframe(df_valid[["Position", "Driver", "Team", "QualifyingTime", "PredictedLap"]])

    # --- BAR CHART ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(df_valid["Driver"], df_valid["PredictedLap"], color="skyblue")
    ax.invert_yaxis()
    ax.set_xlabel("Predicted Race Lap Time (s)")
    ax.set_title("Predicted Race Pace Comparison - Monaco GP 2025")
    st.pyplot(fig)

    # --- IGNORED DRIVERS ---
    excluded = df[df["QualifyingTime"] == 0]["Driver"].tolist()
    if excluded:
        st.info(f"Ignored drivers (no valid qualifying time): {', '.join(excluded)}")

# --- FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align: center;'>© 2025 Monaco GP Predictor | Built using FastF1 + Streamlit</p>", unsafe_allow_html=True)
