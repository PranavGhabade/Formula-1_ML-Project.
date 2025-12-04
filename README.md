# 🏎️ Formula 1 Monaco GP 2025 Race Pace Prediction

This project predicts the **average race lap times for the 2025 Monaco Grand Prix** using historical Formula 1 telemetry data (2022–2024) collected via the **FastF1 Python library**.  
The model applies machine learning techniques to estimate driver performance and race pace based on past qualifying and sector data.

---

## 🚀 Project Overview

The Monaco GP is one of the most challenging circuits in Formula 1 due to its tight corners, narrow track, and strong dependence on qualifying performance.  
This project leverages **data-driven regression modeling** to analyze and predict race performance using driver, team, and lap features.

---

## 🧠 Methodology

1. **Data Collection:** F1 telemetry data (2022–2024) using `FastF1`
2. **Data Cleaning:** Removal of nulls, duplicates, and unit conversion to seconds
3. **Feature Engineering:**  
   - Total Sector Time  
   - Consistency Score  
   - Team Form  
   - Monaco Experience  
4. **Model Training:** `GradientBoostingRegressor` from `scikit-learn`
5. **Validation:** Tested on 2024 data  
   - MAE: 0.82 s  
   - RMSE: 0.99 s  
   - Spearman: 0.57  
6. **Deployment:** Model saved as `.pkl` file and integrated with a `Streamlit` web interface

---

## 📊 Tech Stack

- **Python 3.11+**  
- **Libraries:** FastF1, pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit  
- **Frontend:** Streamlit (interactive driver-wise UI)  
- **Model:** Gradient Boosting Regressor (scikit-learn)

---

## 🧩 Project Structure
├── data
├── notebooks
├── monaco_model.pkl
├── streamlit_app.py
├── README.md


## 🏁 Results

| Metric | Value |
|--------|--------|
| **MAE** | 0.82 s |
| **RMSE** | 0.99 s |
| **Spearman ρ** | 0.57 |
| **Kendall τ** | 0.40 |

**Predicted Top Performers (2025 Monaco GP):**
1. Lando Norris (McLaren)  
2. Charles Leclerc (Ferrari)  
3. Oscar Piastri (McLaren)

## 💡 Future Work

- Include weather and tyre degradation factors  
- Extend the model for multi-track prediction
- Add pit stop strategies into the model
- Explore deep learning models for improved accuracy
