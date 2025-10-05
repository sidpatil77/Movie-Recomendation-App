# app_flask.py
from flask import Flask, request, jsonify, render_template
import os

app = Flask(__name__, template_folder="templates")

# ------------------------------------------------------------
# Lazy loading setup
# ------------------------------------------------------------
recommender = None  # model will be loaded only when needed


def get_recommender():
    """Load recommender only when it's first needed"""
    global recommender
    if recommender is None:
        from recommender.model import MovieRecommender
        recommender = MovieRecommender("data/movies.csv", "data/credits.csv")
    return recommender


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    """Main web page for manual testing"""
    recommendations = []
    error = None
    if request.method == "POST":
        movie = request.form.get("movie", "")
        if movie:
            try:
                recs = get_recommender().recommend(movie)
                recommendations = recs
            except Exception as e:
                error = str(e)
        else:
            error = "Please enter a movie title."
    return render_template("index.html", recommendations=recommendations, error=error)


@app.route("/recommend", methods=["POST"])
def recommend():
    """API endpoint used by Streamlit frontend"""
    data = request.get_json()
    movie = data.get("movie", "")
    try:
        recs = get_recommender().recommend(movie)
        return jsonify({"recommendations": recs})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health")
def health():
    """Health check for Render"""
    return "OK", 200


# ------------------------------------------------------------
# Run app
# ------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
