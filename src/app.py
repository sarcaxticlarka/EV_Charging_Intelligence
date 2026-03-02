import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import os

# ==========================================
# CONFIGURATION & SETUP
# ==========================================
st.set_page_config(
    page_title="EV Charging Intelligence",
    page_icon=":material/bolt:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minor custom CSS to reduce top padding for a cleaner look
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# PATHS & LOADING
# ==========================================
MODEL_PATH = "models/demand_predictor.pkl"
DATA_PATH = "data/processed/ev_charging_processed.csv"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

model = load_model()
df = load_data()

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.title(":material/settings: Control Panel")
    st.write("Configure simulation parameters to forecast EV charging demand.")
    st.divider()

    st.subheader("Time & Schedule")
    hour = st.slider("Hour of Day", 0, 23, 17, help="Select the hour (24-hour format)")
    
    day_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    day_val = st.selectbox("Day of Week", list(day_map.keys()), format_func=lambda x: day_map[x], index=4)
    
    is_weekend = day_val >= 5
    is_peak = (7 <= hour <= 10) or (16 <= hour <= 19)
    if is_peak:
        st.warning("Peak Hours Active", icon=":material/timer:")
        
    st.divider()

    st.subheader("Environment")
    temp_f = st.slider("Temperature (°F)", 0, 120, 85, help="Current temperature")
    precip = st.number_input("Precipitation (mm)", 0.0, 50.0, 0.0, help="Rainfall amount")
    
    weather_opts = sorted(df['weather_category'].unique().tolist()) if df is not None else ['Good', 'Bad', 'Neutral', 'Extreme']
    weather = st.selectbox("Weather Condition", weather_opts)
    
    st.divider()

    st.subheader("Traffic & Economics")
    traffic = st.select_slider(
        "Traffic Congestion", 
        options=[1, 2, 3], 
        value=2, 
        format_func=lambda x: {1: "Low", 2: "Medium", 3: "High"}[x]
    )
    gas_price = st.number_input("Gas Price ($/gallon)", 2.0, 7.0, 4.50, step=0.10)
    
    event_opts = sorted(df['local_event'].unique().tolist()) if df is not None else ['none', 'concert', 'game']
    event = st.selectbox("Local Event", event_opts, format_func=lambda x: str(x).title())
    
    st.divider()

    st.subheader("Station Details")
    city_opts = sorted(df['city'].unique().tolist()) if df is not None else ['San Francisco']
    city = st.selectbox("City", city_opts)
    
    loc_type_opts = sorted(df['location_type'].unique().tolist()) if df is not None else ['Urban Center']
    loc_type = st.selectbox("Location Type", loc_type_opts)
    
    charger_opts = sorted(df['charger_type'].unique().tolist()) if df is not None else ['DC Fast']
    charger = st.selectbox("Charger Type", charger_opts)
    
    st.divider()
    predict_btn_sidebar = st.button("Yo Predict", icon=":material/bolt:", type="primary", use_container_width=True)

# ==========================================
# MAIN DASHBOARD
# ==========================================
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title(":material/battery_charging_full: EV Charging Intelligence")
    st.markdown("Next-generation AI system for predicting electric vehicle charging demand. Optimize infrastructure planning with real-time analytics and intelligent forecasting.")
with header_col2:
    st.write("")
    st.write("")
    predict_btn = st.button("Predict Demand", icon=":material/bolt:", type="primary", use_container_width=True)

if model is None:
    st.error("Model Not Loaded")
    st.write("The prediction model could not be found. Please train the model first by running `python3 src/model_trainer.py`.")
    st.stop()

if predict_btn or predict_btn_sidebar:
    st.divider()
    
    # Prepare input
    input_data = pd.DataFrame({
        'traffic_congestion_index': [traffic],
        'gas_price_per_gallon': [gas_price],
        'temperature_f': [temp_f],
        'precipitation_mm': [precip],
        'hour_of_day': [hour],
        'day_of_week': [day_val],
        'is_weekend': [is_weekend],
        'is_peak_hour': [is_peak],
        'weather_category': [weather],
        'location_type': [loc_type],
        'charger_type': [charger],
        'city': [city],
        'local_event': [event]
    })
    
    # Predict
    prediction = model.predict(input_data)[0]
    prediction = max(0.0, min(1.0, prediction))
    pct = prediction * 100
    
    # Determine level
    if prediction <= 0.3:
        level, color, icon = "Low", "green", ":material/check_circle:"
    elif prediction <= 0.6:
        level, color, icon = "Moderate", "orange", ":material/warning:"
    else:
        level, color, icon = "High", "red", ":material/error:"

    # Navigation Tabs
    tab1, tab2, tab3 = st.tabs([":material/bar_chart: Prediction Overview", ":material/trending_up: Analytics & Trends", ":material/folder: Raw Data"])

    # ---------- TAB 1: DASHBOARD ----------
    with tab1:
        st.subheader(f"Demand Forecast: {level} ({pct:.1f}%)")
        st.progress(prediction, text=f"{icon} Estimated station utilization")
        st.write("")

        # Metrics columns with historical comparisons
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate historical average for delta display if data is available
        hist_avg = 0.0
        if df is not None and not df.empty:
            city_avg = df[df['city'] == city]['utilization_rate'].mean() * 100
            if not np.isnan(city_avg):
                hist_avg = city_avg

        delta_val = pct - hist_avg if hist_avg > 0 else None
        delta_str = f"{delta_val:+.1f}% vs City Avg" if delta_val is not None else None
        
        col1.metric("Predicted Utilization", f"{pct:.1f}%", delta=delta_str, delta_color="inverse")
        col2.metric("Demand Level", level)
        traffic_labels = {1: "Low", 2: "Medium", 3: "High"}
        col3.metric("Traffic Index", traffic_labels[traffic])
        col4.metric("Weather", f"{weather} ({temp_f}°F)")
        
        st.divider()
        st.subheader("Key Demand Drivers")
        st.write("Factors influencing the current prediction:")
        
        drivers_cols = st.columns(3)
        driver_idx = 0
        
        def add_driver(title, impact, desc, icon_str):
            global driver_idx
            col = drivers_cols[driver_idx % 3]
            with col:
                with st.container(border=True):
                    st.markdown(f"**{icon_str} {title}**")
                    st.caption(desc)
                    st.write(f"Impact: `{impact}`")
            driver_idx += 1

        has_drivers = False
        if is_peak:
            add_driver("Peak Hours", "+25%", "Surges typically occur during 7–10 AM and 4–7 PM.", ":material/timer:")
            has_drivers = True
        if traffic == 3:
            add_driver("High Traffic", "+30%", "Congestion leads to more opportunistic charging stops.", ":material/directions_car:")
            has_drivers = True
        if weather == 'Extreme':
            add_driver("Extreme Weather", "+35%", "Temperature extremes significantly drain EV batteries.", ":material/tornado:")
            has_drivers = True
        elif weather == 'Bad':
            add_driver("Adverse Weather", "+15%", "Affects typical charging routines and efficiency.", ":material/rainy:")
            has_drivers = True
        if event != 'none':
            add_driver(f"Local: {str(event).title()}", "+20%", "Events dramatically increase temporary local volume.", ":material/local_activity:")
            has_drivers = True
        if gas_price >= 5.5:
            add_driver("High Fuel Prices", "+18%", "Drives increased adoption and usage of alternatives.", ":material/local_gas_station:")
            has_drivers = True
        if is_weekend:
            add_driver("Weekend Mode", "~10%", "Shifts volume from business districts to retail/leisure.", ":material/event:")
            has_drivers = True
            
        if not has_drivers:
            with st.container(border=True):
                st.markdown("**:material/check_circle: Normal Conditions**")
                st.caption("All parameters are within standard ranges. Demand follows typical patterns.")
                st.write("Impact: `Base`")

    # ---------- TAB 2: ANALYTICS ----------
    with tab2:
        st.subheader("City Level Insights")
        if df is not None and not df.empty:
            city_df = df[df['city'] == city]
            if not city_df.empty:
                chart_cols = st.columns(2)
                
                # Chart 1: Hourly Demand Area Chart
                with chart_cols[0]:
                    hourly_demand = city_df.groupby('hour_of_day')['utilization_rate'].mean().reset_index()
                    hourly_demand['utilization_rate'] = hourly_demand['utilization_rate'] * 100
                    
                    chart = alt.Chart(hourly_demand).mark_area(
                        color="lightblue", 
                        interpolate='monotone',
                        opacity=0.5
                    ).encode(
                        x=alt.X('hour_of_day:Q', title='Hour of Day (0-23)', scale=alt.Scale(domain=[0, 23])),
                        y=alt.Y('utilization_rate:Q', title='Avg Utilization (%)'),
                        tooltip=['hour_of_day', 'utilization_rate']
                    ).properties(title=f"Hourly Demand Pattern in {city}", height=350)
                    
                    chart += alt.Chart(hourly_demand).mark_line(color="#3b82f6").encode(
                        x='hour_of_day:Q', 
                        y='utilization_rate:Q'
                    )
                    st.altair_chart(chart, use_container_width=True)

                # Chart 2: Charger Type Boxplot
                with chart_cols[1]:
                    charger_view = city_df[['charger_type', 'utilization_rate']].copy()
                    charger_view['utilization_rate'] = charger_view['utilization_rate'] * 100
                    
                    boxplot = alt.Chart(charger_view).mark_boxplot(extent='min-max').encode(
                        x=alt.X('charger_type:N', title='Charger Type'),
                        y=alt.Y('utilization_rate:Q', title='Utilization (%)'),
                        color=alt.Color('charger_type:N', legend=None)
                    ).properties(title=f"Demand Distribution by Charger Type in {city}", height=350)
                    st.altair_chart(boxplot, use_container_width=True)
                    
                st.divider()
                
                # Chart 3: Temperature vs Utilization Scatter
                st.subheader("Weather Impact Analysis")
                sample_df = df.sample(min(2000, len(df))) # Sample for performance
                scatter = alt.Chart(sample_df).mark_circle(size=60, opacity=0.6).encode(
                    x=alt.X('temperature_f:Q', title='Temperature (°F)'),
                    y=alt.Y('utilization_rate:Q', title='Utilization', scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color('weather_category:N', title='Weather Condition'),
                    tooltip=['city', 'temperature_f', 'utilization_rate', 'weather_category']
                ).properties(title="System-wide Demand vs Temperature", height=400).interactive()
                st.altair_chart(scatter, use_container_width=True)

            else:
                st.info(f"Not enough historical data available for {city} to display analytics.")
        else:
            st.warning("Historical dataset is missing. Analytics cannot be displayed.")

    # ---------- TAB 3: DATA EXPLORER ----------
    with tab3:
        st.subheader("Input Simulation Parameters")
        st.json(input_data.to_dict(orient='records')[0])
        
        st.subheader("Historical Dataset Sample")
        if df is not None:
            st.dataframe(df.head(100), use_container_width=True)
        else:
            st.warning("Historical data file not found.")

else:
    st.info("Please configure the parameters in the sidebar and click **Predict Demand** to continue.", icon=":material/arrow_back:")
    
    st.markdown("### Welcome to the AI Dashboard")
    st.write("This application predicts electric vehicle charging station utilization based on a simulated Random Forest model. Use the sidebar to tweak real-world variables like traffic, weather, and time of day to see how they impact local charging demand.")