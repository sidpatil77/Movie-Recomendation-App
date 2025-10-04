"""
recommender/model.py

Robust MovieRecommender that:
- Attempts to load data/movies.csv and data/credits.csv
- If not present, writes small sample CSVs so app still runs
- Handles both TMDB JSON-string columns and simpler CSV formats
- Builds 'tags', vectorizes using CountVectorizer and computes cosine similarity
- Exposes recommend(title, top_n=5)
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import warnings

SAMPLE_MOVIES_CSV = """id,title,overview,genres,keywords
1,Avatar,"A marine on an alien planet", "[{'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}]", "[{'id': 101, 'name':'alien'}]"
2,The Matrix,"A computer hacker learns reality is a simulation", "[{'id': 878, 'name': 'Science Fiction'}]", "[{'id': 102, 'name':'ai'}]"
3,Interstellar,"Explorers travel through a wormhole", "[{'id': 12, 'name': 'Adventure'}, {'id': 18, 'name':'Drama'}]", "[{'id': 103, 'name':'space'}]"
4,Inception,"A thief who steals corporate secrets by dream-sharing", "[{'id': 878, 'name': 'Science Fiction'}]", "[{'id': 104, 'name':'dream'}]"
"""

SAMPLE_CREDITS_CSV = """movie_id,cast,crew
1,"[{'cast_id':1,'name':'Sam Worthington'},{'cast_id':2,'name':'Zoe Saldana'}]","[{'job':'Director','name':'James Cameron'}]"
2,"[{'cast_id':1,'name':'Keanu Reeves'},{'cast_id':2,'name':'Carrie-Anne Moss'}]","[{'job':'Director','name':'Lana Wachowski'}]"
3,"[{'cast_id':1,'name':'Matthew McConaughey'},{'cast_id':2,'name':'Anne Hathaway'}]","[{'job':'Director','name':'Christopher Nolan'}]"
4,"[{'cast_id':1,'name':'Leonardo DiCaprio'},{'cast_id':2,'name':'Joseph Gordon-Levitt'}]","[{'job':'Director','name':'Christopher Nolan'}]"
"""

def ensure_data_files(movies_path, credits_path):
    """If the user's CSVs don't exist, create small sample files so app runs."""
    os.makedirs(os.path.dirname(movies_path) or ".", exist_ok=True)
    if not os.path.isfile(movies_path):
        with open(movies_path, "w", encoding="utf-8") as f:
            f.write(SAMPLE_MOVIES_CSV)
    if not os.path.isfile(credits_path):
        with open(credits_path, "w", encoding="utf-8") as f:
            f.write(SAMPLE_CREDITS_CSV)

class MovieRecommender:
    def __init__(self, movies_path="data/movies.csv", credits_path="data/credits.csv"):
        # Make sure data exists (sample fallback ensures app will run)
        ensure_data_files(movies_path, credits_path)

        # Read CSVs
        self.movies_df = pd.read_csv(movies_path)
        self.credits_df = pd.read_csv(credits_path)

        # Prepare data and build model
        self._prepare()

    # ---------------------------
    # Utilities to parse different formats
    # ---------------------------
    def _safe_ast(self, text):
        """Safely parse stringified lists/dicts from TMDB columns. Returns list or empty list."""
        if pd.isna(text):
            return []
        if isinstance(text, list):
            return text
        try:
            return ast.literal_eval(text)
        except Exception:
            # If it's a simple pipe-separated string like 'Action|Adventure'
            if isinstance(text, str) and "|" in text:
                return [{"name": part.strip()} for part in text.split("|")]
            return []

    def _get_names_from_list(self, parsed):
        """Given a parsed list of dicts / strings, return a list of names."""
        names = []
        if not parsed:
            return names
        # parsed might be list of dicts
        for item in parsed:
            if isinstance(item, dict):
                # TMDB style: { "id": 28, "name": "Action" } or cast dict with 'name'
                if "name" in item:
                    names.append(str(item["name"]).replace(" ", ""))
            else:
                # item might be a string
                names.append(str(item).replace(" ", ""))
        return names

    def _parse_cast(self, text):
        parsed = self._safe_ast(text)
        # parsed is list of dicts with key 'name' -> return up to first 3 cast member names
        names = []
        for item in parsed:
            if isinstance(item, dict) and "name" in item:
                names.append(item["name"].replace(" ", ""))
            elif isinstance(item, str):
                names.append(item.replace(" ", ""))
            if len(names) >= 3:
                break
        return names

    def _parse_crew_director(self, text):
        parsed = self._safe_ast(text)
        for item in parsed:
            if isinstance(item, dict) and item.get("job") == "Director":
                return [item.get("name", "").replace(" ", "")]
        return [""]  # keep list for concatenation

    # ---------------------------
    # Data preparation and model building
    # ---------------------------
    def _prepare(self):
        # Merge. TMDB movies file uses id, credits uses movie_id
        # First try merge on 'id' and 'movie_id', else try merge on 'title' if present.
        try:
            self.df = self.movies_df.merge(self.credits_df, left_on="id", right_on="movie_id")
        except Exception:
            # fallback merge on title
            if "title" in self.movies_df.columns and "title" in self.credits_df.columns:
                self.df = self.movies_df.merge(self.credits_df, on="title")
            else:
                # naive: just concatenate columns where possible
                self.df = self.movies_df.copy()
                for col in self.credits_df.columns:
                    if col not in self.df.columns:
                        self.df[col] = self.credits_df[col]

        # Ensure columns exist
        for c in ["title", "overview", "genres", "keywords", "cast", "crew"]:
            if c not in self.df.columns:
                self.df[c] = ""

        # Fill NaNs
        self.df["overview"] = self.df["overview"].fillna("")
        self.df["genres"] = self.df["genres"].fillna("")
        self.df["keywords"] = self.df["keywords"].fillna("")
        self.df["cast"] = self.df["cast"].fillna("")
        self.df["crew"] = self.df["crew"].fillna("")

        # Build usable columns
        # genres and keywords: parse list of dicts or pipe-separated string
        self.df["genres_list"] = self.df["genres"].apply(lambda x: self._get_names_from_list(self._safe_ast(x)))
        self.df["keywords_list"] = self.df["keywords"].apply(lambda x: self._get_names_from_list(self._safe_ast(x)))
        self.df["cast_list"] = self.df["cast"].apply(self._parse_cast)
        self.df["director_list"] = self.df["crew"].apply(self._parse_crew_director)

        # overview -> words
        self.df["overview_words"] = self.df["overview"].apply(lambda x: [w.replace(" ", "") for w in str(x).split()])

        # Combine into tags
        self.df["tags"] = self.df.apply(
            lambda row: " ".join(row["overview_words"] + row["genres_list"] + row["keywords_list"] + row["cast_list"] + row["director_list"]),
            axis=1
        )

        # Lowercase
        self.df["tags"] = self.df["tags"].str.lower()

        # Vectorize
        self.cv = CountVectorizer(max_features=5000, stop_words="english")
        try:
            self.vectors = self.cv.fit_transform(self.df["tags"]).toarray()
        except Exception as e:
            warnings.warn(f"Vectorization failed: {e}. Falling back to empty vectors.")
            self.vectors = np.zeros((len(self.df), 1))

        # Compute similarity matrix
        try:
            self.similarity = cosine_similarity(self.vectors)
        except Exception:
            self.similarity = np.zeros((len(self.df), len(self.df)))

        # Lowercase title helper
        self.df["title_lower"] = self.df["title"].astype(str).str.lower()

    def recommend(self, movie_title, top_n=5):
        """Return list of top_n recommended movie titles for the given movie_title.
        Raises ValueError if movie not found."""
        if not isinstance(movie_title, str) or movie_title.strip() == "":
            raise ValueError("Please provide a valid movie title string.")

        movie_title_lower = movie_title.strip().lower()

        if movie_title_lower not in self.df["title_lower"].values:
            # Try partial match (find titles that contain the input)
            candidates = self.df[self.df["title_lower"].str.contains(movie_title_lower)]
            if not candidates.empty:
                idx = candidates.index[0]
            else:
                raise ValueError(f"Movie '{movie_title}' not found in the dataset.")
        else:
            idx = self.df[self.df["title_lower"] == movie_title_lower].index[0]

        distances = list(enumerate(self.similarity[idx]))
        # sort by similarity and exclude itself
        distances = sorted(distances, key=lambda x: x[1], reverse=True)
        results = []
        for i, score in distances[1: top_n + 1 + 1]:  # a bit of extra incase of duplicates
            results.append(self.df.iloc[i]["title"])
            if len(results) >= top_n:
                break
        return results

