import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import os
import json

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
    /* Agent chat styling */
    .agent-node-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 8px;
        color: #e0e0e0;
    }
    .agent-node-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #00d2ff;
        margin-bottom: 4px;
    }
    .agent-node-status {
        font-size: 0.8rem;
        color: #a0a0a0;
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
# APP MODE SELECTOR (Sidebar Top)
# ==========================================
with st.sidebar:
    st.title(":material/settings: Control Panel")
    st.divider()
    app_mode = st.radio(
        "Select Mode",
        [":material/bar_chart: ML Dashboard", ":material/smart_toy: AI Trip Planner"],
        index=0,
        help="Switch between the traditional ML dashboard and the AI-powered trip planner."
    )
    st.divider()


# ##########################################################
#  MODE 1: TRADITIONAL ML DASHBOARD  (Original app.py)
# ##########################################################
if "ML Dashboard" in app_mode:
    with st.sidebar:
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


# ##########################################################
#  MODE 2: AGENTIC AI TRIP PLANNER  (LangGraph)
# ##########################################################
elif "Trip Planner" in app_mode:

    # ── Header ───────────────────────────────────────────
    st.title(":material/smart_toy: Smart EV Trip & Charging Planner")
    st.markdown(
        "Ask me about your upcoming EV trip in natural language. "
        "I'll use an **Agentic AI workflow (LangGraph)** to parse your request, "
        "run an ML model, fetch live data, and synthesize a personalised charging itinerary."
    )

    # ── Architecture Expander ────────────────────────────
    with st.expander(":material/account_tree: How the Agent Works — LangGraph Architecture", expanded=False):
        st.markdown("""
**This planner uses a 4-node LangGraph state machine:**

| Node | Role | Technology |
|------|------|-----------|
| 🧠 **Reasoner** | Parses your natural language query into structured trip parameters | Groq API |
| 🤖 **ML Tool** | Runs the trained Random Forest model to predict station utilization & wait times | Scikit-Learn Pipeline |
| 🌐 **API Tool** | Fetches live traffic conditions, weather advisories, and electricity pricing | External APIs (Simulated) |
| 📝 **Synthesizer** | Combines all data into a personalized, actionable trip itinerary | Groq API |

```
[User Query] → 🧠 Reasoner → 🤖 ML Tool → 🌐 API Tool → 📝 Synthesizer → [Trip Plan]
```
        """)

    st.divider()

    # ── Check for API Key ────────────────────────────────
    api_key = os.environ.get("GROQ_API_KEY", "")
    
    with st.sidebar:
        st.subheader(":material/key: API Configuration")
        api_key_input = st.text_input(
            "Groq API Key",
            value=api_key,
            type="password",
            help="Get a free key from https://console.groq.com/keys"
        )
        if api_key_input:
            os.environ["GROQ_API_KEY"] = api_key_input

        st.divider()
        st.subheader(":material/help: Example Queries")
        example_queries = [
            "I'm driving to New York this Friday at 5 PM. I have a Level 2 charger. Where and when should I charge to avoid traffic and high fees?",
            "Planning a weekend trip to Chicago, leaving Saturday morning. DC Fast charger. What's the best charging strategy?",
            "Need to get to San Francisco by 9 AM Monday. It's going to rain. Should I charge tonight or on the way?",
        ]
        for i, q in enumerate(example_queries):
            if st.button(f"Try Example {i+1}", key=f"example_{i}", use_container_width=True):
                st.session_state["agent_query"] = q

    # ── Chat Interface ───────────────────────────────────
    # Initialize chat history
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []

    # Display chat history
    for msg in st.session_state.agent_messages:
        with st.chat_message(msg["role"], avatar="🚗" if msg["role"] == "user" else "⚡"):
            st.markdown(msg["content"])
            # Show node trace if available
            if "trace" in msg:
                with st.expander(":material/account_tree: Agent Reasoning Trace", expanded=False):
                    for step in msg["trace"]:
                        st.markdown(f"**{step.get('node', 'Unknown')}** — {step.get('status', '')}")
                        if "data" in step:
                            st.json(step["data"])

    # Get user input
    default_query = st.session_state.pop("agent_query", None)
    user_input = st.chat_input("Describe your EV trip... (e.g., 'Driving to New York Friday at 5 PM with Level 2 charger')")
    
    query = default_query or user_input

    if query:
        # Check API key
        if not os.environ.get("GROQ_API_KEY"):
            st.error("Please enter your Groq API Key in the sidebar to use the AI Trip Planner.")
            st.stop()

        # Display user message
        st.session_state.agent_messages.append({"role": "user", "content": query})
        with st.chat_message("user", avatar="🚗"):
            st.markdown(query)

        # Run the agent
        with st.chat_message("assistant", avatar="⚡"):
            with st.status("🔄 Running Agentic Workflow...", expanded=True) as status:
                try:
                    # Import the agent
                    from agent import run_planner
                    
                    st.write("🧠 **Node 1:** Parsing trip parameters with LLM...")
                    result = run_planner(query)

                    # Show trace in status
                    for step in result.get("node_trace", []):
                        st.write(f"{step.get('node', '')} — {step.get('status', '')}")

                    if result.get("error"):
                        status.update(label="⚠️ Agent encountered an issue", state="error")
                    else:
                        status.update(label="✅ Trip plan generated!", state="complete")

                except ImportError as e:
                    result = {
                        "final_response": f"⚠️ **Import Error:** Could not load the agent module. Make sure all dependencies are installed:\n```\npip install -r requirements.txt\n```\nError: `{str(e)}`",
                        "node_trace": [],
                        "error": str(e),
                    }
                    status.update(label="❌ Agent failed to load", state="error")

                except Exception as e:
                    result = {
                        "final_response": f"⚠️ **Error:** {str(e)}",
                        "node_trace": [],
                        "error": str(e),
                    }
                    status.update(label="❌ Agent error", state="error")

            # Display final response
            st.divider()
            final_response = result.get("final_response", "No response generated.")
            st.markdown(final_response)

            # Show detailed trace
            trace = result.get("node_trace", [])
            if trace:
                with st.expander(":material/account_tree: Agent Reasoning Trace (Full Details)", expanded=False):
                    for step in trace:
                        with st.container(border=True):
                            st.markdown(f"**{step.get('node', 'Unknown')}**")
                            st.caption(step.get("status", ""))
                            if "data" in step:
                                st.json(step["data"])

            # Show ML prediction metrics if available
            ml_pred = result.get("ml_prediction")
            if ml_pred:
                st.divider()
                st.subheader(":material/analytics: ML Model Insights")
                m1, m2, m3 = st.columns(3)
                m1.metric("Station Utilization", f"{ml_pred.get('utilization_pct', 0)}%")
                m2.metric("Est. Wait Time", f"{ml_pred.get('estimated_wait_minutes', 0)} min")
                m3.metric("Demand Level", ml_pred.get('demand_level', 'N/A'))

            # Save to history
            st.session_state.agent_messages.append({
                "role": "assistant",
                "content": final_response,
                "trace": trace,
            })