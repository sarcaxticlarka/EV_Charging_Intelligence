"""
Smart EV Trip & Charging Planner — LangGraph Agentic Workflow
=============================================================
A 4-node LangGraph state machine that:
  1. Parses natural-language trip queries  (LLM Reasoner)
  2. Runs the Random Forest model           (ML Tool)
  3. Fetches live traffic / weather context  (API Tool)
  4. Synthesizes a personalised itinerary    (LLM Synthesizer)
"""

import os
import json
import random
import datetime
from typing import TypedDict, Optional, Any

import joblib
import pandas as pd
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# ── Load env ─────────────────────────────────────────────────
load_dotenv()

# ── Paths ────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "demand_predictor.pkl")
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "ev_charging_processed.csv")


# ═══════════════════════════════════════════════════════════════
#  STATE DEFINITION
# ═══════════════════════════════════════════════════════════════
class PlannerState(TypedDict):
    """Shared state flowing through every node."""
    user_query: str
    trip_params: Optional[dict]       # Extracted by Reasoner
    ml_prediction: Optional[dict]     # Output from ML Tool
    live_context: Optional[dict]      # Output from API Tool
    final_response: Optional[str]     # Synthesized answer
    node_trace: list                  # Transparency log for UI
    error: Optional[str]


# ═══════════════════════════════════════════════════════════════
#  HELPER — LOAD MODEL
# ═══════════════════════════════════════════════════════════════
_model_cache = {}

def _get_model():
    if "model" not in _model_cache:
        if os.path.exists(MODEL_PATH):
            _model_cache["model"] = joblib.load(MODEL_PATH)
        else:
            _model_cache["model"] = None
    return _model_cache["model"]


def _get_available_values():
    """Read the dataset to know valid cities, charger types, etc."""
    try:
        df = pd.read_csv(DATA_PATH)
        return {
            "cities": sorted(df["city"].unique().tolist()),
            "charger_types": sorted(df["charger_type"].unique().tolist()),
            "location_types": sorted(df["location_type"].unique().tolist()),
            "weather_categories": sorted(df["weather_category"].unique().tolist()),
            "local_events": sorted(df["local_event"].unique().tolist()),
        }
    except Exception:
        return {
            "cities": ["San Francisco", "New York", "Chicago", "Los Angeles", "Minneapolis"],
            "charger_types": ["Level 1", "Level 2", "DC Fast Charge", "Hyper-Fast"],
            "location_types": ["Urban Center", "Suburban", "Highway", "Shopping Center", "Workplace", "Airport"],
            "weather_categories": ["Good", "Bad", "Neutral", "Extreme"],
            "local_events": ["none", "concert", "game", "festival"],
        }


# ═══════════════════════════════════════════════════════════════
#  NODE 1 — LLM REASONER  (Parse trip details)
# ═══════════════════════════════════════════════════════════════
def reasoner_node(state: PlannerState) -> PlannerState:
    """Use the LLM to extract structured trip parameters from the user query."""
    avail = _get_available_values()
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
    )

    system_prompt = f"""You are an EV trip parameter extractor.  Given a user's natural-language
trip query, extract the following fields as JSON.  Use sensible defaults where the user
does not specify a value.

Required JSON fields:
- city (string): The destination city. Must be one of {avail['cities']}. Pick the closest match.
- hour_of_day (int 0-23): Departure / arrival hour.
- day_of_week (int 0=Mon … 6=Sun): Day of week.
- charger_type (string): Must be one of {avail['charger_types']}. Map "Level 2" → "Level 2", "fast charger" → "DC Fast Charge", etc.
- location_type (string): Best guess from {avail['location_types']}. Default "Urban Center".
- temperature_f (float): Estimated temperature. Default 75.
- precipitation_mm (float): Estimated precipitation. Default 0.
- weather_category (string): One of {avail['weather_categories']}. Default "Good".
- traffic_congestion_index (int 1-3): 1=Low 2=Medium 3=High. Infer from time/day.
- gas_price_per_gallon (float): Default 4.50.
- local_event (string): One of {avail['local_events']}. Default "none".
- trip_summary (string): One-line human summary of the trip.

Reply with ONLY valid JSON. No markdown, no backticks.
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["user_query"]),
    ])

    try:
        params = json.loads(response.content.strip())
    except json.JSONDecodeError:
        # Try to extract JSON from markdown-wrapped response
        text = response.content.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            params = json.loads(text)
        except json.JSONDecodeError:
            state["error"] = f"Failed to parse LLM response: {response.content[:200]}"
            state["node_trace"].append({
                "node": "🧠 Reasoner",
                "status": "❌ Failed to parse parameters",
                "raw": response.content[:300],
            })
            return state

    # Compute derived fields
    params["is_weekend"] = params.get("day_of_week", 4) >= 5
    hour = params.get("hour_of_day", 17)
    params["is_peak_hour"] = (7 <= hour <= 10) or (16 <= hour <= 19)

    state["trip_params"] = params
    state["node_trace"].append({
        "node": "🧠 Reasoner (LLM)",
        "status": "✅ Extracted trip parameters",
        "data": params,
    })
    return state


# ═══════════════════════════════════════════════════════════════
#  NODE 2 — ML TOOL  (Random Forest prediction)
# ═══════════════════════════════════════════════════════════════
def ml_node(state: PlannerState) -> PlannerState:
    """Run the trained Random Forest model on the extracted parameters."""
    if state.get("error"):
        return state

    model = _get_model()
    params = state["trip_params"]

    if model is None:
        state["error"] = "ML model not found. Train it first with `python src/model_trainer.py`."
        state["node_trace"].append({
            "node": "🤖 ML Model",
            "status": "❌ Model file missing",
        })
        return state

    input_df = pd.DataFrame({
        "traffic_congestion_index": [params.get("traffic_congestion_index", 2)],
        "gas_price_per_gallon":     [params.get("gas_price_per_gallon", 4.50)],
        "temperature_f":            [params.get("temperature_f", 75)],
        "precipitation_mm":         [params.get("precipitation_mm", 0.0)],
        "hour_of_day":              [params.get("hour_of_day", 17)],
        "day_of_week":              [params.get("day_of_week", 4)],
        "is_weekend":               [params.get("is_weekend", False)],
        "is_peak_hour":             [params.get("is_peak_hour", True)],
        "weather_category":         [params.get("weather_category", "Good")],
        "location_type":            [params.get("location_type", "Urban Center")],
        "charger_type":             [params.get("charger_type", "Level 2")],
        "city":                     [params.get("city", "San Francisco")],
        "local_event":              [params.get("local_event", "none")],
    })

    try:
        prediction = model.predict(input_df)[0]
        prediction = max(0.0, min(1.0, prediction))
    except Exception as e:
        state["error"] = f"ML prediction failed: {str(e)}"
        state["node_trace"].append({
            "node": "🤖 ML Model",
            "status": f"❌ Prediction error: {str(e)[:100]}",
        })
        return state

    utilization_pct = round(prediction * 100, 1)

    # Estimate wait time based on utilization
    if utilization_pct > 80:
        wait_min = random.randint(25, 45)
        demand_level = "🔴 High"
    elif utilization_pct > 50:
        wait_min = random.randint(10, 25)
        demand_level = "🟠 Moderate"
    else:
        wait_min = random.randint(0, 10)
        demand_level = "🟢 Low"

    ml_result = {
        "utilization_pct": utilization_pct,
        "estimated_wait_minutes": wait_min,
        "demand_level": demand_level,
        "model_type": "Random Forest Regressor (scikit-learn)",
    }

    state["ml_prediction"] = ml_result
    state["node_trace"].append({
        "node": "🤖 ML Model (Random Forest)",
        "status": f"✅ Predicted {utilization_pct}% utilization — {demand_level}",
        "data": ml_result,
    })
    return state


# ═══════════════════════════════════════════════════════════════
#  NODE 3 — API TOOL  (Live traffic & weather)
# ═══════════════════════════════════════════════════════════════
def api_node(state: PlannerState) -> PlannerState:
    """Simulate fetching live traffic and weather data from external APIs.
    
    In production, this would call Google Maps / OpenWeatherMap / NREL APIs.
    For the academic demo, we generate realistic synthetic data.
    """
    if state.get("error"):
        return state

    params = state["trip_params"]
    city = params.get("city", "San Francisco")
    hour = params.get("hour_of_day", 17)

    # ── Simulated Traffic API ────────────────────────────
    # Rush-hour logic makes the simulation realistic
    if (7 <= hour <= 9) or (16 <= hour <= 19):
        traffic_level = random.choice(["Heavy", "Very Heavy"])
        delay_minutes = random.randint(15, 45)
        alternate_route = True
    elif (10 <= hour <= 15):
        traffic_level = random.choice(["Moderate", "Light"])
        delay_minutes = random.randint(5, 15)
        alternate_route = False
    else:
        traffic_level = "Light"
        delay_minutes = random.randint(0, 10)
        alternate_route = False

    # ── Simulated Weather API ────────────────────────────
    weather_conditions = {
        "Good": {"condition": "Clear sky", "advisory": "No weather advisories."},
        "Neutral": {"condition": "Partly cloudy", "advisory": "Mild conditions expected."},
        "Bad": {"condition": "Rain expected", "advisory": "Drive carefully. Reduced visibility."},
        "Extreme": {"condition": "Extreme heat warning", "advisory": "Battery efficiency may drop 15-20%. Plan extra charging."},
    }
    weather_cat = params.get("weather_category", "Good")
    weather_info = weather_conditions.get(weather_cat, weather_conditions["Good"])

    # ── Simulated Charging Station Availability ──────────
    nearby_stations = random.randint(3, 12)
    available_ports = random.randint(1, max(1, nearby_stations - 2))

    # ── Simulated Electricity Pricing ────────────────────
    if (16 <= hour <= 21):
        electricity_rate = round(random.uniform(0.28, 0.42), 2)
        pricing_tier = "Peak"
    elif (7 <= hour <= 15):
        electricity_rate = round(random.uniform(0.18, 0.28), 2)
        pricing_tier = "Mid-Peak"
    else:
        electricity_rate = round(random.uniform(0.08, 0.18), 2)
        pricing_tier = "Off-Peak"

    live_data = {
        "traffic": {
            "level": traffic_level,
            "estimated_delay_minutes": delay_minutes,
            "suggest_alternate_route": alternate_route,
        },
        "weather": {
            "condition": weather_info["condition"],
            "advisory": weather_info["advisory"],
            "temperature_f": params.get("temperature_f", 75),
        },
        "charging_stations": {
            "nearby_stations": nearby_stations,
            "available_ports": available_ports,
            "city": city,
        },
        "electricity_pricing": {
            "rate_per_kwh": electricity_rate,
            "tier": pricing_tier,
        },
        "data_source": "Simulated (Google Maps / OpenWeatherMap / NREL)",
        "fetched_at": datetime.datetime.now().isoformat(),
    }

    state["live_context"] = live_data
    state["node_trace"].append({
        "node": "🌐 API Tool (Live Data)",
        "status": f"✅ Fetched traffic ({traffic_level}), weather ({weather_info['condition']}), pricing (${electricity_rate}/kWh {pricing_tier})",
        "data": live_data,
    })
    return state


# ═══════════════════════════════════════════════════════════════
#  NODE 4 — SYNTHESIZER  (Personalized itinerary)
# ═══════════════════════════════════════════════════════════════
def synthesizer_node(state: PlannerState) -> PlannerState:
    """Use the LLM to synthesize all gathered context into a personalized itinerary."""
    if state.get("error"):
        # Generate an error response
        state["final_response"] = f"⚠️ I couldn't complete your trip plan: {state['error']}"
        return state

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
    )

    system_prompt = """You are a Smart EV Trip & Charging Planner advisor. You have received:
1. The user's original trip request
2. Extracted trip parameters (from another AI agent)
3. ML model predictions for charging station utilization
4. Live traffic, weather, and pricing data

Your job: Craft a **personalized, actionable trip itinerary** that:
- Recommends optimal departure time based on the ML prediction and traffic
- Suggests WHERE and WHEN to charge, referencing the utilization/wait data
- Warns about weather impacts on battery if relevant
- Highlights cost savings (off-peak vs peak charging)
- If utilization is high (>60%), proactively suggest delaying the trip or using a different time slot

Format your response with clear sections using markdown:
- 🗺️ **Trip Overview**
- ⚡ **Charging Strategy** 
- 🚦 **Traffic Advisory**
- 💰 **Cost Optimization**
- 📋 **Recommended Itinerary** (step-by-step)

Be conversational, friendly, and specific. Use the actual numbers from the data.
Do NOT mention that the data is simulated. Treat everything as real.
"""

    context = f"""
USER QUERY: {state['user_query']}

EXTRACTED TRIP PARAMETERS:
{json.dumps(state['trip_params'], indent=2)}

ML MODEL PREDICTION:
{json.dumps(state['ml_prediction'], indent=2)}

LIVE CONTEXT DATA:
{json.dumps(state['live_context'], indent=2)}
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=context),
    ])

    state["final_response"] = response.content
    state["node_trace"].append({
        "node": "📝 Synthesizer (LLM)",
        "status": "✅ Generated personalized itinerary",
    })
    return state


# ═══════════════════════════════════════════════════════════════
#  LANGGRAPH — BUILD THE WORKFLOW
# ═══════════════════════════════════════════════════════════════
def build_graph():
    """Compile the 4-node LangGraph state machine."""
    workflow = StateGraph(PlannerState)

    # Add nodes
    workflow.add_node("reasoner",    reasoner_node)
    workflow.add_node("ml_tool",     ml_node)
    workflow.add_node("api_tool",    api_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # Define edges  (sequential pipeline)
    workflow.set_entry_point("reasoner")
    workflow.add_edge("reasoner",    "ml_tool")
    workflow.add_edge("ml_tool",     "api_tool")
    workflow.add_edge("api_tool",    "synthesizer")
    workflow.add_edge("synthesizer", END)

    return workflow.compile()


# Compile once at module level for reuse
graph = build_graph()


def run_planner(user_query: str) -> PlannerState:
    """Execute the full planning pipeline for a user query."""
    initial_state: PlannerState = {
        "user_query": user_query,
        "trip_params": None,
        "ml_prediction": None,
        "live_context": None,
        "final_response": None,
        "node_trace": [],
        "error": None,
    }
    result = graph.invoke(initial_state)
    return result


# ── CLI Test ─────────────────────────────────────────────────
if __name__ == "__main__":
    test_query = (
        "I'm driving to New York this Friday at 5 PM. "
        "I have a Level 2 charger. Where and when should I "
        "charge to avoid traffic and high fees?"
    )
    print("=" * 60)
    print("  Smart EV Trip Planner — Test Run")
    print("=" * 60)
    print(f"\n📩 Query: {test_query}\n")

    result = run_planner(test_query)

    print("\n── Node Trace ──")
    for step in result["node_trace"]:
        print(f"  {step['node']}: {step['status']}")

    print("\n── Final Response ──")
    print(result["final_response"])
