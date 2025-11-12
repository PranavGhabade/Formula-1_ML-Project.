import fastf1
import pandas as pd
import os


os.makedirs("f1_cache", exist_ok=True)

# Enable cache
fastf1.Cache.enable_cache("f1_cache")

# Config
YEAR = 2024
ROUND = 8  # Monaco 2024
SESSIONS = ["FP1", "FP2", "FP3", "Q", "R"]

all_laps = []

for session_type in SESSIONS:
    print(f"Loading {session_type}...")
    session = fastf1.get_session(YEAR, ROUND, session_type)
    session.load()

    laps = session.laps.copy()
    
    # Convert time columns to seconds
    time_cols = ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]
    for col in time_cols:
        if col in laps.columns:
            laps[f"{col}_s"] = laps[col].dt.total_seconds()

    # Label session type
    laps["Session"] = session_type

    # Driver name
    laps["DriverName"] = laps["Driver"]

    # Keep only useful columns
    keep_cols = [
        "Driver", "DriverName", "Team", "Session",
        "LapNumber", "Stint", "Compound",
        "LapTime_s", "Sector1Time_s", "Sector2Time_s", "Sector3Time_s"
    ]
    
    # Filter existing columns
    keep_cols = [col for col in keep_cols if col in laps.columns]

    all_laps.append(laps[keep_cols])

# Concatenate all sessions
full_data = pd.concat(all_laps, ignore_index=True)

# Drop empty rows
full_data.dropna(subset=["LapTime_s"], inplace=True)

# Save to CSV
output_file = "f1_monaco_2024_full_dataset.csv"
full_data.to_csv(output_file, index=False)

print(f"Single dataset saved as: {output_file}")
print("Shape:", full_data.shape)
