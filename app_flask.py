from flask import Flask, request, jsonify
from recommender.model import MovieRecommender
import os

app = Flask(__name__)
recommender = MovieRecommender("data/movies.csv", "data/credits.csv")

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    movie = data.get("movie", "")
    try:
        recs = recommender.recommend(movie)
        return jsonify({"recommendations": recs})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/health")
def health():
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
