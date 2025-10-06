import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self, movies_path, credits_path):
        self.movies_path = movies_path
        self.credits_path = credits_path
        self.df = None
        self.cosine_sim = None
        self._prepare()

    def _safe_ast(self, text):
        """Safely parse stringified lists (from CSVs)."""
        try:
            if isinstance(text, str) and text.strip():
                return ast.literal_eval(text)
        except Exception:
            return []
        return []

    def _get_names_from_list(self, data):
        """Extract 'name' fields from list of dicts."""
        if isinstance(data, list):
            return [d.get("name", "") for d in data if isinstance(d, dict)]
        return []

    def _prepare(self):
        """Load and prepare movie data (optimized for small memory)."""
        print("📂 Loading datasets...")
        movies = pd.read_csv(self.movies_path)
        credits = pd.read_csv(self.credits_path)

        print("🔗 Merging datasets...")
        df = movies.merge(credits, on="title")

        print("🧹 Cleaning and processing data...")
        # Use only a subset to stay within memory limits
        df = df.head(5000)

        df["cast_list"] = df["cast"].apply(lambda x: self._get_names_from_list(self._safe_ast(x)))
        df["crew_list"] = df["crew"].apply(lambda x: self._get_names_from_list(self._safe_ast(x)))
        df["keywords_list"] = df["keywords"].apply(lambda x: self._get_names_from_list(self._safe_ast(x)))
        df["genres_list"] = df["genres"].apply(lambda x: self._get_names_from_list(self._safe_ast(x)))

        df["tags"] = df["cast_list"] + df["crew_list"] + df["keywords_list"] + df["genres_list"]
        df["tags"] = df["tags"].apply(lambda x: " ".join(x))

        print("📊 Vectorizing text data...")
        cv = CountVectorizer(max_features=5000, stop_words="english")
        vectors = cv.fit_transform(df["tags"]).toarray()

        print("🧮 Calculating similarity matrix...")
        self.cosine_sim = cosine_similarity(vectors)
        self.df = df.reset_index(drop=True)
        print("✅ Data prepared successfully.")

    def recommend(self, movie_title):
        """Recommend similar movies."""
        if movie_title not in self.df["title"].values:
            raise ValueError(f"Movie '{movie_title}' not found in dataset.")

        idx = self.df[self.df["title"] == movie_title].index[0]
        distances = list(enumerate(self.cosine_sim[idx]))
        movies_sorted = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
        recommendations = [self.df.iloc[i[0]].title for i in movies_sorted]
        return recommendations
