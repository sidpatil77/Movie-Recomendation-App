# app_flask.py - Flask application for deployment
from flask import Flask, render_template, request
from recommender.model import MovieRecommender
import os

def create_app():
    """Factory function for creating the Flask app."""
    app = Flask(__name__, template_folder="templates")

    # Load model at startup
    recommender = MovieRecommender("data/movies.csv", "data/credits.csv")

    @app.route("/", methods=["GET", "POST"])
    def index():
        recommendations = []
        error = None
        if request.method == "POST":
            movie = request.form.get("movie", "")
            if movie:
                try:
                    recommendations = recommender.recommend(movie)
                except Exception as e:
                    error = str(e)
            else:
                error = "Please enter a movie title."
        return render_template("index.html", recommendations=recommendations, error=error)

    @app.route("/health")
    def health():
        return "OK", 200

    return app


# --- The critical part for Render ---
# Render launches gunicorn automatically using the Procfile
app = create_app()

if __name__ == "__main__":
    # Local run only (Render uses gunicorn)
    port = int(os.environ.get("PORT", 10000))  # Force 10000 fallback port
    app.run(host="0.0.0.0", port=port, debug=False)
