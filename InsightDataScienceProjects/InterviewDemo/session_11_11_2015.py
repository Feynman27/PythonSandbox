# coding: utf-8
import pandas as pd
df_movies = pd.read_csv('../data/ml-latest-small/movies.csv')
print df_movies.head()
df_ratings = pd.read_csv('../data/ml-latest-small/ratings.csv',usecols=range(3))
print df_ratings.head()
df_imdb = pd.read_csv('../data/ml-latest-small/links.csv',usecols=range(2))
print df_imdb.head()

df = pd.merge(df_movies,df_ratings,left_on='movieId',right_on='movieId',how='outer')
df = pd.merge(df,df_imdb,left_on='movieId',right_on='movieId',how='outer')
print df.head()
