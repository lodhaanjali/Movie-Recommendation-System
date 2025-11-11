from flask import Flask, request, render_template_string, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# main.py
# Simple Flask website for a movie recommendation system using NLP (TF-IDF) + cosine similarity.
# Requires: flask, scikit-learn, pandas
# Install: pip install flask scikit-learn pandas


app = Flask(__name__)

# Small example movie dataset (replace or expand with a real dataset)
movies = [
    {"id": 0, "title": "Star Voyagers", "genres": "Sci-Fi Adventure", "overview": "A crew explores distant planets and faces strange phenomena."},
    {"id": 1, "title": "Love in Autumn", "genres": "Romance Drama", "overview": "Two strangers meet during a rainy autumn and learn to heal."},
    {"id": 2, "title": "Quantum Heist", "genres": "Action Thriller", "overview": "A team uses experimental tech to pull off a near-impossible robbery."},
    {"id": 3, "title": "Silent Forest", "genres": "Horror Mystery", "overview": "Visitors get lost in a forest where reality bends and secrets whisper."},
    {"id": 4, "title": "Chef's Journey", "genres": "Comedy Drama", "overview": "A young chef travels the country to rediscover family recipes and meaning."},
    {"id": 5, "title": "Galactic Wars", "genres": "Sci-Fi Action", "overview": "Interstellar factions clash in a battle for a valuable energy source."},
    {"id": 6, "title": "Midnight Detective", "genres": "Crime Noir", "overview": "A private detective uncovers corruption while searching for a missing heir."},
    {"id": 7, "title": "Ocean Echoes", "genres": "Documentary", "overview": "An exploration of ocean life and the communities that depend on it."},
    {"id": 8, "title": "The Last Marathon", "genres": "Sports Drama", "overview": "An aging runner prepares for one final race that might redeem his past."},
    {"id": 9, "title": "Parallel Lives", "genres": "Sci-Fi Romance", "overview": "Two people living in parallel realities try to find a bridge between them."},
]

df = pd.DataFrame(movies)

# Build text corpus (combine overview and genres)
df["text"] = (df["title"] + " " + df["genres"] + " " + df["overview"]).fillna("")

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["text"])

# Precompute cosine similarity matrix for movie-to-movie recommendations
cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Helper: get top N similar movie indices given a movie id (excluding itself)
def recommend_by_movie(movie_id, top_n=5):
    if movie_id not in df["id"].values:
        return []
    idx = df.index[df["id"] == movie_id][0]
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i for i, score in sim_scores[1: top_n + 1]]  # skip first (itself)
    return df.iloc[top_indices].to_dict(orient="records")

# Helper: get top N similar movies given a free-text query
def recommend_by_text(query, top_n=5):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_indices = sims.argsort()[::-1][:top_n]
    return df.iloc[top_indices].to_dict(orient="records")

# Simple HTML template embedded to keep one-file app
TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Movie Recommender</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
<div class="container py-5">
  <h1 class="mb-4">Movie Recommendation System</h1>

  <div class="card mb-4">
    <div class="card-body">
      <form method="post" action="{{ url_for('recommend') }}">
        <div class="mb-3">
          <label class="form-label">Find similar movies to...</label>
          <select name="movie_id" class="form-select">
            <option value="">-- choose a movie --</option>
            {% for m in movies %}
            <option value="{{m.id}}" {% if selected_movie and selected_movie==m.id %}selected{% endif %}>{{m.title}} ({{m.genres}})</option>
            {% endfor %}
          </select>
        </div>

        <div class="mb-3">
          <label class="form-label">Or enter a description / mood / keywords</label>
          <input name="query" class="form-control" placeholder="e.g., space adventure with battles and heroes" value="{{ query|default('') }}">
        </div>

        <div class="d-flex gap-2">
          <button class="btn btn-primary" type="submit">Get Recommendations</button>
          <a class="btn btn-outline-secondary" href="{{ url_for('index') }}">Reset</a>
        </div>
      </form>
    </div>
  </div>

  {% if results %}
  <h4>Recommendations</h4>
  <div class="row">
    {% for r in results %}
    <div class="col-md-6">
      <div class="card mb-3">
        <div class="card-body">
          <h5 class="card-title">{{ r.title }}</h5>
          <h6 class="card-subtitle mb-2 text-muted">{{ r.genres }}</h6>
          <p class="card-text">{{ r.overview }}</p>
        </div>
      </div>
    </div>
    {% endfor %}
  </div>
  {% endif %}

  <hr>
  <p class="text-muted small">This is a demo using TF-IDF + cosine similarity. Replace the dataset and expand preprocessing for better results.</p>
</div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(TEMPLATE, movies=movies, results=None, selected_movie=None, query="")

@app.route("/recommend", methods=["POST"])
def recommend():
    movie_id = request.form.get("movie_id", "")
    query = request.form.get("query", "").strip()
    results = []

    try:
        if movie_id:
            results = recommend_by_movie(int(movie_id), top_n=5)
        if query:
            # If both provided, combine: text-based results override or you can merge logic
            results = recommend_by_text(query, top_n=5)
    except Exception:
        results = []

    # If nothing provided, redirect to home
    if not movie_id and not query:
        return redirect(url_for("index"))

    return render_template_string(TEMPLATE, movies=movies, results=results, selected_movie=int(movie_id) if movie_id else None, query=query)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)