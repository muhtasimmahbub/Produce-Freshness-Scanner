import streamlit as st
import requests
import os

API_URL = os.getenv(
    "API_URL", "https://produce-freshness-scanner.onrender.com/predict-image"
)

st.set_page_config(
    page_title="Produce Freshness Scanner",
    page_icon="🍎",
    layout="centered",
)

st.title("🍎 Produce Freshness Scanner")
st.markdown("Upload an image or take a photo to evaluate produce freshness.")

st.divider()

source = st.radio("Select Image Source:", ["File Upload", "Camera Capture"], horizontal=True)

uploaded_file = None
if source == "File Upload":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("Take a picture")

if uploaded_file is not None:
    st.image(uploaded_file, caption="Input Frame", use_container_width=True)

    if st.button("Analyze Freshness", type="primary", use_container_width=True):
        files = {
            "file": (
                uploaded_file.name if hasattr(uploaded_file, "name") else "capture.jpg",
                uploaded_file.getvalue(),
                uploaded_file.type if hasattr(uploaded_file, "type") else "image/jpeg",
            )
        }

        try:
            with st.spinner("Analyzing frame..."):
                response = requests.post(API_URL, files=files, timeout=10)

            if response.status_code == 200:
                data = response.json()
                status = data.get("status") or data.get("label", "UNKNOWN")
                confidence = data.get("confidence", 0.0)
                resolution = data.get("resolution", "N/A")

                st.subheader("Analysis Results")
                col1, col2 = st.columns(2)

                with col1:
                    if status in ["FRESH", "Fresh"]:
                        st.success(f"**Status:** {status}")
                    else:
                        st.error(f"**Status:** {status}")

                with col2:
                    st.metric(label="Model Confidence", value=f"{confidence * 100:.1f}%")

                st.caption(f"Resolution: {resolution}")

            else:
                st.error(f"API Error ({response.status_code}): {response.text}")

        except requests.exceptions.Timeout:
            st.error("⌛ Request timed out. FastAPI took longer than 5 seconds to respond.")
        except requests.exceptions.ConnectionError:
            st.error("🔴 Could not connect to FastAPI. Ensure `uvicorn src.api:app --reload` is running on port 8000.")
        except requests.exceptions.RequestException as e:
            st.error(f"⚠️ Network error: {e}")