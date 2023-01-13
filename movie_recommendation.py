import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load in your datasets
df1 = pd.read_csv('tmdb_5000_credits.csv')
df2 = pd.read_csv('tmdb_5000_movies.csv')

# Merge and preprocess the data as you did in your previous code
df1.columns = ['id','tittle','cast','crew']
df2 = df2.merge(df1,on='id')
df2['release_dates'] = pd.to_datetime(df2['release_date']).dt.year
df2['release_dates'].fillna(2003, inplace=True)
df2['release_dates'] = df2['release_dates'].apply(lambda x: int(x))

# Define the TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
df2['overview'] = df2['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(df2['overview'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a reverse index of movie titles
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()

# Define the recommendation function
def get_recommendations(title,cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df2[['title', 'release_dates','director','runtime', 'original_language', 'vote_average','genres']].iloc[movie_indices]

# Use Streamlit to create the UI
st.title("Movie Recommendation System")

# Get the user's input
title = st.text_input("Enter the title of a movie:")

# Show the recommendations
if title:
    recommendations = get_recommendations(title)
    st.dataframe(recommendations)
else:
    st.write("Enter a movie title to get recommendations.")
