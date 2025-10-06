# app_flask.py - Flask backend for Movie Recommendation App (Render-compatible)
from flask import Flask, request, jsonify
from recommender.model import MovieRecommender
import os
import traceback
import sys

app = Flask(__name__)

# --- Lazy load the recommender model ---
recommender = None

def get_recommender():
    """Load the recommender model only when first requested."""
    global recommender
    if recommender is None:
        print("🧠 Loading Movie Recommender model...")
        try:
            recommender = MovieRecommender("data/movies.csv", "data/credits.csv")
            print("✅ Model loaded successfully.")
        except Exception as e:
            print("❌ Failed to load model:", e)
            traceback.print_exc(file=sys.stdout)
            raise e
    return recommender


# --- Health check route (for Render) ---
@app.route("/health")
def health():
    return "OK", 200


# --- Main recommendation API ---
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json(force=True)
        movie = data.get("movie", "").strip()

        if not movie:
            return jsonify({"error": "No movie title provided."}), 400

        model = get_recommender()
        recommendations = model.recommend(movie)

        return jsonify({"recommendations": recommendations}), 200

    except Exception as e:
        print("🔥 ERROR:", e)
        traceback.print_exc(file=sys.stdout)
        return jsonify({"error": str(e)}), 500


# --- Root route (optional simple message) ---
@app.route("/")
def home():
    return jsonify({"message": "Movie Recommender API is running!"}), 200


# --- App startup (Render detects port via $PORT env var) ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting Flask app on port {port}")
    app.run(host="0.0.0.0", port=port)
