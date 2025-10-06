# app_flask.py
from flask import Flask, request, jsonify
from recommender.model import MovieRecommender
import threading
import os

app = Flask(__name__)

# Global variable for the recommender (lazy loaded)
recommender = None
load_lock = threading.Lock()

def get_recommender():
    """Lazily load the recommender model (only once)."""
    global recommender
    if recommender is None:
        with load_lock:
            if recommender is None:  # Double-check (thread-safe)
                print("🔄 Loading recommender model...")
                recommender = MovieRecommender("data/movies.csv", "data/credits.csv")
                print("✅ Recommender model loaded.")
    return recommender

@app.route("/", methods=["GET"])
def home():
    return "🎬 Movie Recommender API is running!", 200

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json() or {}
    movie = data.get("movie", "").strip()

    if not movie:
        return jsonify({"error": "Missing 'movie' field"}), 400

    try:
        recs = get_recommender().recommend(movie)
        return jsonify({"recommendations": recs})
    except Exception as e:
        print(f"❌ Error while recommending: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
