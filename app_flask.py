# app_flask.py - Flask application for deployment
from flask import Flask, render_template, request
from recommender.model import MovieRecommender
import os

app = Flask(__name__, template_folder="templates")
# Load model at startup (fast for sample; for very large datasets consider lazy load)
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

# Health check (useful for Render)
@app.route("/health")
def health():
    return "OK", 200

if __name__ == "__main__":
    # Only for local debugging; Render will run via gunicorn
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
