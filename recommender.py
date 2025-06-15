import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import  cosine_similarity
import ast

df=pd.read_csv("data/movies.csv")

def parse_genre(str_genre):
    try:
        genres=ast.literal_eval(str_genre)
        return " ".join(genre["name"] for genre in genres)
    except:
        return ""

df["combined"]=df["overview"].fillna('')+' '+df["genres"].apply(parse_genre)

cv=CountVectorizer(max_features=5000,stop_words='english')
vectors=cv.fit_transform(df["combined"])

similarity=cosine_similarity(vectors)

def recommend_movie(movie):
    if movie not in df["title"].values:
        print("movie not found")
    
    idx=df[df["title"]==movie].index[0]
    distances= similarity[idx]
    movie_list=sorted(list(enumerate(distances)),key= lambda x : x[1],reverse=True)[1:6]
    return [df.iloc[i[0]] ["title"] for i in movie_list]
        
if __name__ == "__main__":
    movie=input("Enter the movie title")
    movies=recommend_movie(movie)
    print(movies)
    




