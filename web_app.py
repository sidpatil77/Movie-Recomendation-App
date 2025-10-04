# web_app.py - Streamlit UI
import streamlit as st
from recommender.model import MovieRecommender

st.set_page_config(page_title="Movie Recommender", layout="centered")

@st.cache_resource
def load_model():
    return MovieRecommender("data/movies.csv", "data/credits.csv")

recommender = load_model()

st.title("🎬 Movie Recommender")
st.write("Type a movie title and get similar suggestions (content-based).")

movie = st.text_input("Movie title (e.g., Inception)")

if st.button("Get Recommendations"):
    if not movie:
        st.warning("Please enter a movie title.")
    else:
        try:
            recs = recommender.recommend(movie)
            st.success("Top recommendations:")
            for i, r in enumerate(recs, 1):
                st.write(f"**{i}.** {r}")
        except Exception as e:
            st.error(str(e))
