import pickle
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load your dataset
df = pd.read_csv("clustered_df.csv")
overview_tfidf = pickle.load(open("overview_tfidf.pkl",'rb'))


def recommend_movies(title, df, num_recommendations=6):
    if title not in df['Series_Title'].values:
        return "Movie not found in dataset."

    cluster_label = df[df['Series_Title'] == title]['Cluster'].values[0]
    cluster_movies = df[df['Cluster'] == cluster_label]
    movie_vector = overview_tfidf[df[df['Series_Title'] == title].index[0]]
    similarities = cosine_similarity(movie_vector, overview_tfidf[cluster_movies.index]).flatten()
    similar_indices = similarities.argsort()[-(num_recommendations + 1):-1][::-1]
    recommendations = cluster_movies.iloc[similar_indices][['Series_Title', 'Overview', 'IMDB_Rating', 'Poster_Link']]
    return recommendations.reset_index(drop=True)


# Streamlit UI
st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("🎬 Movie Recommendation System")

movie_list = df['Series_Title'].sort_values().unique()
selected_movie = st.selectbox("Select a Movie", movie_list)

if st.button("Recommend"):
    output = recommend_movies(selected_movie, df)

    if isinstance(output, str):
        st.error(output)
    else:
        for i in range(0, len(output), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(output):
                    row = output.iloc[i + j]
                    with cols[j]:
                        st.markdown(f"### {row['Series_Title']}")
                        st.image(row['Poster_Link'], width=150)
                        st.write(f"**IMDB Rating:** {row['IMDB_Rating']}")
                        st.write(f"**Overview:** {row['Overview'][:40]}....")
                        st.markdown("---")

