# web_app.py - Streamlit frontend connected to Flask backend
import streamlit as st
import requests
import os

# -----------------------------------------------------------------------------
# ✅ CONFIGURATION
# -----------------------------------------------------------------------------
# Get the Flask API URL from environment variables (set in Render)
API_URL = os.environ.get("API_URL", "http://localhost:5000")  # local fallback
RECOMMEND_ENDPOINT = f"{API_URL}/recommend"

st.set_page_config(page_title="Movie Recommender", layout="centered")

# -----------------------------------------------------------------------------
# 🎬 UI
# -----------------------------------------------------------------------------
st.title("🎬 Movie Recommender")
st.write("Type a movie title and get similar suggestions (powered by Flask API).")

movie = st.text_input("🎞️ Movie title (e.g., Inception)")

if st.button("Get Recommendations"):
    if not movie.strip():
        st.warning("Please enter a movie title.")
    else:
        try:
            # -----------------------------------------------------------------------------
            # 🌐 Call Flask backend
            # -----------------------------------------------------------------------------
            with st.spinner("Fetching recommendations..."):
                response = requests.post(RECOMMEND_ENDPOINT, json={"movie": movie})
            
            # -----------------------------------------------------------------------------
            # 📦 Handle Response
            # -----------------------------------------------------------------------------
            if response.status_code == 200:
                data = response.json()
                recs = data.get("recommendations", [])
                if recs:
                    st.success("Top Recommendations:")
                    for i, r in enumerate(recs, 1):
                        st.write(f"**{i}.** {r}")
                else:
                    st.info("No recommendations found for that movie.")
            else:
                st.error(f"Server returned an error: {response.status_code}")
                st.text(response.text)
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to API: {e}")
