import streamlit as st
import pandas as pd
import numpy as np
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

from ast import literal_eval
features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(literal_eval)

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

# Returns the list top 3 elements or entire list; whichever is more.
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []

# Define new director, cast, genres and keywords features that are in a suitable form.
df2['director'] = df2['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(get_list)
    # Define the recommendation function
    
    
def get_recommendations(title,cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df2[['title', 'release_dates','director','runtime', 'original_language', 'vote_average','genres']].iloc[movie_indices]

# Use Streamlit to create the UI
# Add CSS styles
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon=":movie_camera:",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
   
    <style>
    /* Add custom styles here */
    /* You can use CSS to change the appearance of your app */
    /* For example, you can change the font, color, and size of text */
    /* You can also add background colors, borders, and other styles */
    body {
        font-family: Arial, sans-serif;
    }
    h1, h2, h3, h4 {
        color: #1f4287;
    }
    .stMarkdown h1 {
        font-size: 2.5em;
        margin: 0.5em 0;
    }
    .stMarkdown h2 {
        font-size: 2em;
        margin: 0.5em 0;
    }
    .stMarkdown h3 {
        font-size: 1.5em;
        margin: 0.5em 0;
    }
    .stMarkdown p {
        font-size: 1.2em;
        margin: 0.5em 0;
    }
    /* You can also add custom styles to specific elements */
    /* For example, you can add a background color to the input field */
    /* and change the color of the submit button */
    .stTextInput {
        background-color: #f2f2f2;
    }
    .stButton {
        background-color: #1f4287;
        color: white;
    }
    </style>
    
)

st.title("Movie Recommendation System")

# Get the user's input
title = st.text_input("Enter the title of a movie:")

# Show the recommendations
if title:
    recommendations = get_recommendations(title)
    st.dataframe(recommendations)
else:
    st.write("Enter a movie title to get recommendations.")
