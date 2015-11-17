#!/usr/bin/python

'''
Build item-based collaborative filter by finding
items that are similar to ones user has bought
'''

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import scipy.stats as stats
import pandas as pd

if __name__ == "__main__":

    ## read in dataframes
    df_movies = pd.read_csv('../../data/ml-latest-small/movies.csv')
    #print df_movies.head()
    df_ratings = pd.read_csv('../../data/ml-latest-small/ratings.csv',usecols=range(3))
    #print df_ratings.head()
    df_imdb = pd.read_csv('../../data/ml-latest-small/links.csv',usecols=range(2))
    #print df_imdb.head()

    ## merge by movieId
    df = pd.merge(df_movies,df_ratings,left_on='movieId',right_on='movieId',how='outer')
    df = pd.merge(df,df_imdb,left_on='movieId',right_on='movieId',how='outer')
    print df.head()

    df_2 = pd.DataFrame(df.title.str.split('(',1).tolist(),columns=['title','relDate'])
    df_2['relDate'] = df_2['relDate'].apply(lambda x: str(x).replace(')', ''))
    df = pd.merge(df,df_2,how='outer',left_index=True,right_index=True)
    df = df.drop('title_x',1)
    df.rename(columns={'title_y':'title'},inplace=True)
    df['relDate'] = df['relDate'].convert_objects(convert_numeric=True)
    print df.head()

    df_new = df[df['relDate']>2010]
    print df_new.head()

    import sys
    #print top scores for given user
    null_values = 0
    user_name = raw_input("Enter username: ")
    while (user_name not in str(df['userId'])) :
        print 'Cannot locate username. Please retry.'
        user_name = raw_input("Enter username: ")

