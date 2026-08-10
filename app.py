import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="Produce Freshness Scanner",
    page_icon="🍎",
    layout="centered"
)

st.title("🍎 Produce Freshness Scanner")
st.markdown("Adjust the color hue and texture smoothness metrics below to evaluate produce freshness.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    color = st.slider(
        "Color Hue Value",
        min_value=0.0,
        max_value=10.0,
        value=8.2,
        step=0.1,
        help="8.0+ typical for fresh produce; low values indicate decay/discoloration."
    )

with col2:
    texture = st.slider(
        "Texture Smoothness",
        min_value=0.0,
        max_value=10.0,
        value=7.1,
        step=0.1,
        help="7.5+ indicates smooth skin; low values indicate bruising or degradation."
    )

st.divider()

if st.button("Analyze Freshness", type="primary", use_container_width=True):
    payload = {"color": color, "texture": texture}
    
    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            status = data.get("status")
            confidence = data.get("confidence")
            
            st.subheader("Analysis Results")
            if status == "FRESH":
                st.success(f"**Status:** {status}")
            else:
                st.error(f"**Status:** {status}")
                
            st.metric(label="Model Confidence", value=f"{confidence}%")
        else:
            st.error(f"API Error ({response.status_code}): {response.text}")
            
    except requests.exceptions.Timeout:
        st.error("⌛ Request timed out. FastAPI took longer than 5 seconds to respond.")
    except requests.exceptions.ConnectionError:
        st.error("🔴 Could not connect to FastAPI. Ensure `uvicorn src.api:app --reload` is running on port 8000.")
    except requests.exceptions.RequestException as e:
        st.error(f"⚠️ Network error: {e}")