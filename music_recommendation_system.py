import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
songs = pd.read_csv('/content/songdata.csv')
songs = songs.sample(n=5000).drop('link', axis=1).reset_index(drop=True)
songs['text'] = songs['text'].str.replace(r'\n', '')

# Compute TF-IDF matrix
tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
lyrics_matrix = tfidf.fit_transform(songs['text'])
cosine_similarities = cosine_similarity(lyrics_matrix)

# Create similarity dictionary
similarities = {}
for i in range(len(cosine_similarities)):
    similar_indices = cosine_similarities[i].argsort()[:-50:-1]
    similarities[songs['song'].iloc[i]] = [(cosine_similarities[i][x], songs['song'][x], songs['artist'][x]) for x in similar_indices][1:]

# Content-Based Recommender Class
class ContentBasedRecommender:
    def __init__(self, matrix):
        self.matrix_similar = matrix

    def _print_message(self, song, recom_song):
        rec_items = len(recom_song)
        print(f'The {rec_items} recommended songs for {song} are:')
        for i in range(rec_items):
            print(f"Number {i+1}:")
            print(f"{recom_song[i][1]} by {recom_song[i][2]} with {round(recom_song[i][0], 3)} similarity score")
            print("--------------------")

    def recommend(self, recommendation):
        song = recommendation['song']
        number_songs = recommendation['number_songs']
        recom_song = self.matrix_similar.get(song, [])[:number_songs]
        self._print_message(song=song, recom_song=recom_song)

# Create recommender instance
recommender = ContentBasedRecommender(similarities)

# Example recommendations
recommendation1 = {
    "song": songs['song'].iloc[10],
    "number_songs": 4
}
recommender.recommend(recommendation1)

recommendation2 = {
    "song": songs['song'].iloc[120],
    "number_songs": 4
}
recommender.recommend(recommendation2)

recommendation3 = {
    "song": songs['song'].iloc[250],
    "number_songs": 4
}
recommender.recommend(recommendation3)

recommendation4 = {
    "song": songs['song'].iloc[350],
    "number_songs": 4
}
recommender.recommend(recommendation4)
