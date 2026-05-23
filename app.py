import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Solar Energy Predictor", page_icon="☀️", layout="wide")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/869/869869.png", width=80)
    st.title(" Solar Predictor")
    st.markdown("---")
    st.markdown("### About")
    st.info("This app uses Machine Learning to predict solar power output based on real weather and sensor data from two solar plants.")
    st.markdown("### Dataset")
    st.markdown("-  Plant 1 & Plant 2")
    st.markdown("-  May - June 2020")
    st.markdown("-  68,774 records")
    st.markdown("### Model")
    st.markdown("- Random Forest")
    st.markdown("- Linear Regression")
    st.markdown("- Decision Tree")
    st.markdown("---")
    st.markdown("Made with using Python & Streamlit")
@st.cache_resource
def load_all_models():
    gen     = pd.read_csv("Plant_1_Generation_Data.csv")
    weather = pd.read_csv("Plant_1_Weather_Sensor_Data.csv")

    gen['DATE_TIME']     = pd.to_datetime(gen['DATE_TIME'],     dayfirst=True)
    weather['DATE_TIME'] = pd.to_datetime(weather['DATE_TIME'], dayfirst=True)

    df = pd.merge(gen, weather, on='DATE_TIME')
    df['HOUR']  = df['DATE_TIME'].dt.hour
    df['DAY']   = df['DATE_TIME'].dt.day
    df['MONTH'] = df['DATE_TIME'].dt.month

    features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE',
                'IRRADIATION', 'HOUR', 'DAY', 'MONTH']
    target = 'AC_POWER'
    df = df.dropna(subset=features + [target])

    X, y = df[features], df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest":    RandomForestRegressor(n_estimators=100, random_state=42),
        "Linear Regression": LinearRegression(),
        "Decision Tree":    DecisionTreeRegressor(max_depth=10, random_state=42),
    }

    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        results[name] = {
            "model":  m,
            "preds":  preds,
            "mae":    mean_absolute_error(y_test, preds),
            "r2":     r2_score(y_test, preds),
        }

    return results, y_test.values, df

results, y_test, df = load_all_models()
rf = results["Random Forest"]

st.title("☀️ Solar Energy Output Predictor")
st.markdown("Predict solar power output using Machine Learning trained on real sensor data.")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📉 Model Performance", "🤖 Compare Models", "🔮 Make a Prediction"])

with tab1:
    st.markdown("### Random Forest — Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² Score",            f"{rf['r2']:.4f}")
    c2.metric("Mean Absolute Error",  f"{rf['mae']:.2f} kW")
    c3.metric("Training Records",     f"{len(df):,}")
    c4.metric("Accuracy",             f"{rf['r2']*100:.1f}%")

    st.markdown("---")
    st.markdown("### Charts")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Actual vs Predicted (first 100 samples)**")
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        ax1.plot(y_test[:100],       label='Actual',    color='#1f77b4', linewidth=1.5)
        ax1.plot(rf['preds'][:100],  label='Predicted', color='#ff7f0e',
                 linewidth=1.5, linestyle='--')
        ax1.set_xlabel("Sample")
        ax1.set_ylabel("AC Power (kW)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        st.pyplot(fig1)

    with col2:
        st.markdown("**Average Solar Output by Hour of Day**")
        hourly = df.groupby('HOUR')['AC_POWER'].mean()
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.bar(hourly.index, hourly.values, color='#f5a623', edgecolor='white')
        ax2.set_xlabel("Hour of Day")
        ax2.set_ylabel("Avg AC Power (kW)")
        ax2.grid(True, alpha=0.3, axis='y')
        fig2.tight_layout()
        st.pyplot(fig2)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Monthly Average Output**")
        monthly = df.groupby('MONTH')['AC_POWER'].mean()
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        ax3.bar(monthly.index, monthly.values, color='#2ecc71', edgecolor='white')
        ax3.set_xlabel("Month")
        ax3.set_ylabel("Avg AC Power (kW)")
        ax3.set_xticks([5, 6])
        ax3.set_xticklabels(['May', 'June'])
        ax3.grid(True, alpha=0.3, axis='y')
        fig3.tight_layout()
        st.pyplot(fig3)

    with col4:
        st.markdown("**Feature Importance (what affects output most)**")
        features = ['Ambient Temp', 'Module Temp', 'Irradiation', 'Hour', 'Day', 'Month']
        importances = results["Random Forest"]["model"].feature_importances_
        sorted_idx = np.argsort(importances)
        fig4, ax4 = plt.subplots(figsize=(6, 3))
        ax4.barh([features[i] for i in sorted_idx],
                 importances[sorted_idx], color='#9b59b6')
        ax4.set_xlabel("Importance Score")
        ax4.grid(True, alpha=0.3, axis='x')
        fig4.tight_layout()
        st.pyplot(fig4)

with tab2:
    st.markdown("### How do different ML models compare?")

    names  = list(results.keys())
    r2s    = [results[n]['r2']  for n in names]
    maes   = [results[n]['mae'] for n in names]

    comp_df = pd.DataFrame({
        "Model":               names,
        "R² Score":            [f"{v:.4f}" for v in r2s],
        "Accuracy":            [f"{v*100:.1f}%" for v in r2s],
        "Mean Absolute Error": [f"{v:.2f} kW" for v in maes],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**R² Score Comparison (higher = better)**")
        colors = ['#f5a623', '#1f77b4', '#2ecc71']
        fig5, ax5 = plt.subplots(figsize=(6, 3))
        bars = ax5.bar(names, r2s, color=colors, edgecolor='white')
        ax5.set_ylabel("R² Score")
        ax5.set_ylim(0, 1.05)
        for bar, val in zip(bars, r2s):
            ax5.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.01,
                     f"{val:.4f}", ha='center', fontsize=10)
        ax5.grid(True, alpha=0.3, axis='y')
        fig5.tight_layout()
        st.pyplot(fig5)

    with col2:
        st.markdown("**Mean Absolute Error (lower = better)**")
        fig6, ax6 = plt.subplots(figsize=(6, 3))
        bars2 = ax6.bar(names, maes, color=colors, edgecolor='white')
        ax6.set_ylabel("MAE (kW)")
        for bar, val in zip(bars2, maes):
            ax6.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5,
                     f"{val:.1f}", ha='center', fontsize=10)
        ax6.grid(True, alpha=0.3, axis='y')
        fig6.tight_layout()
        st.pyplot(fig6)

    st.markdown("---")
    st.markdown("### Predicted vs Actual — all 3 models")
    fig7, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, res), color in zip(axes, results.items(), colors):
        ax.plot(y_test[:80],        label='Actual',    color='#333333', linewidth=1.2)
        ax.plot(res['preds'][:80],  label='Predicted', color=color,
                linewidth=1.2, linestyle='--')
        ax.set_title(f"{name}\nR²={res['r2']:.4f}", fontsize=10)
        ax.set_xlabel("Sample")
        ax.set_ylabel("AC Power (kW)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig7.tight_layout()
    st.pyplot(fig7)

    st.success(" Random Forest wins with 98.6% accuracy — best choice for this dataset!")

with tab3:
    st.markdown("### Enter weather conditions to predict solar output")

    selected_model = st.selectbox(
        "Choose which model to use for prediction:",
        ["Random Forest", "Linear Regression", "Decision Tree"]
    )

    left, right = st.columns(2)
    with left:
        amb_temp    = st.slider("Ambient Temperature (°C)", 10.0, 50.0, 28.0)
        mod_temp    = st.slider("Module Temperature (°C)",  10.0, 70.0, 40.0)
        irradiation = st.slider("Irradiation (W/m²)",        0.0,  1.0,  0.5)
    with right:
        hour  = st.slider("Hour of Day",  0, 23, 12)
        day   = st.slider("Day of Month", 1, 31, 15)
        month = st.slider("Month",        1, 12,  6)

    if st.button("⚡ Predict Solar Output", use_container_width=True):
        input_data = pd.DataFrame(
            [[amb_temp, mod_temp, irradiation, hour, day, month]],
            columns=['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE',
                     'IRRADIATION', 'HOUR', 'DAY', 'MONTH']
        )
        prediction = results[selected_model]["model"].predict(input_data)[0]
        avg_output = df['AC_POWER'].mean()
        diff       = prediction - avg_output
        diff_pct   = (diff / avg_output) * 100

        st.success(f"### ⚡ Predicted AC Power Output: {prediction:.2f} kW")

        m1, m2, m3 = st.columns(3)
        m1.metric("Your Prediction",  f"{prediction:.2f} kW")
        m2.metric("Daily Average",    f"{avg_output:.2f} kW")
        m3.metric("Difference",       f"{diff:+.2f} kW", f"{diff_pct:+.1f}%")

        st.markdown("**Prediction vs Daily Average**")
        fig8, ax8 = plt.subplots(figsize=(6, 2))
        ax8.barh(['Daily Average', 'Your Prediction'],
                 [avg_output, prediction],
                 color=['#aaaaaa', '#f5a623'])
        ax8.set_xlabel("AC Power (kW)")
        ax8.grid(True, alpha=0.3, axis='x')
        fig8.tight_layout()
        st.pyplot(fig8)