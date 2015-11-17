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


def sim_pearson(index,x_1,x_2):
    n = len(index)
    if n==0:return 0
    # sum of ratings for each user
    sum1=np.sum(x_1)
    sum2=np.sum(x_2)

    # sum of squares x_i*x_i
    sumsq1 = np.sum(pow(x_1,2))
    sumsq2 = np.sum(pow(x_2,2))

    # sum of off-diagonal products x_i*y_i
    sumsq12 = np.sum(x_1*x_2)

    # calculate pearson correlation coeff
    '''
        E[XY] - E[X]E[Y]
        ---------------
        sqrt(E[X^2]-(E[X])^2)*sqrt(E[Y^2]-(E[Y])^2)

        which for a sample may be written as

            {n\sum x_i y_i - \sum x_i \sum y_i}
        r = ----------------
            \sqrt{n \sum x_i^2-(\sum x_i)^2} \sqrt{n \sum y_i^2-(\sum y_i)^2}
    '''
    num = sumsq12-(sum1*sum2/n)
    den =  np.sqrt((sumsq1-(pow(sum1,2)/n) )*(sumsq2-(pow(sum2,2)/n) ) )
    r=num/den
    return r

def sim_distance(index,x_1,x_2):

    ## index = list of usernames
    ## x_i = vector of reviews for movie i
    n = len(index)
    if n==0:return 0

    sum_of_squares=0

    # euclidiean distance of all movies: (x_1-y_1)^2 + (x_2-y_2)^2 + ...
    # from vector of ratings from all users
    sum_of_squares = np.sum(pow(x_1-x_2,2))

    # return similarity score
    return 1.0/(1.0+np.sqrt(sum_of_squares))

## Compute similarities between movie ratings by a user.
## If items are rated similarly by a user,
## the items are treated as similar
def top_matches(df,item,n=5,similarity=sim_pearson,max_iter=10):
    # list of tuples
    scores=[]

    count = 0
    ## iterate over all movie pairs
    for i, item_comb in enumerate(combinations(df['movieId'].unique(),2)):

        ## use only those pairs with the current movie of interest
        if(item in item_comb ):
            ## Drop NaN from these two columns
            #df_pair = df[(df['movieId']==item_comb[0]) | (df['movieId']==item_comb[1])].dropna()
            if(item_comb[0]!=item) : other_item=item_comb[0]
            else : other_item=item_comb[1]

            #df_item = df['rating'][df['movieId']==item].reset_index(drop=True)
            #df_item = pd.DataFrame({item:df['rating'][df['movieId']==item]})
            #df_otheritem = df['rating'][df['movieId']==other_item].reset_index(drop=True)
            #dict_pair = {item:df['rating'][df['movieId']==item], other_item:df['rating'][df['movieId']==other_item]}
            #df_pair = pd.DataFrame(dict_pair)
            #print df_pair.head()
            #df_item = pd.DataFrame({item:df['rating'][df['movieId']==item]}).reset_index(drop=True)
            #df_otheritem = pd.DataFrame({other_item:df['rating'][df['movieId']==other_item]}).reset_index(drop=True)
            df_item = df[['userId','rating']][df['movieId']==item].reset_index(drop=True)
            df_otheritem = df[['userId','rating']][df['movieId']==other_item].reset_index(drop=True)
            df_pair = pd.merge(df_item,df_otheritem,how='outer',left_on='userId',right_on='userId')
            df_pair = df_pair.dropna()
            print df_pair.head()

            # store similarity score
            scores.append((similarity(df_pair['userId'],df_pair['rating_x'],df_pair['rating_y']),other_item))

    ## Sort similarity scores in descending order (i.e. most similar items first)
    scores.sort()
    scores.reverse()
    # return top n scores
    return scores[0:n]

def calculate_similar_items(df,n=10):
    # Create dictionary of items that stores other similarly rated items
    result={}
    print 'Computing similarity scores. This may take a minute...'
    # iterate over each movie item
    for i,item in enumerate(df['movieId'].unique()):
        # calculate similarity scores between this movie item and other
        # movie items and store as a tuple
        result[item] = top_matches(df,item,n=n,similarity=sim_distance)

    print 'Done.'
    return result

def get_recommendations(df,itemScores,usr):
    # list of tuples
    rankings=[]
    totalSim={}
    scores={}

    ## invert the dataframe so that
    ## movies are indexes and users are columns
    #df_t = df.T
    ## Ratings for only this user
    df_usr = df[['rating','userId','movieId','title']][df['userId']==int(usr)]
    df_usr = df_usr.dropna()
    df_usr = df_usr.reset_index(drop=True)
    print df_usr['movieId'].unique()

    # Loop over movie items rated by this user
    for index,item in enumerate(df_usr['movieId'].unique()):
        ## make sure key exists
        if(item in itemScores.keys()):
            # Loop over items similar to this item
            for (sim_score,other_item) in itemScores[item]:
                print sim_score
                print other_item
                # ignore item if user has alread rated it
                if other_item == item: continue
                # unrated item is scored by weighting the rating of this item
                # with the similarity score between this item and the current unrated item
                rating=df_usr['rating'][index]
                scores.setdefault(other_item,0.0)
                # weighted sum for unrated item
                scores[other_item]+=rating * sim_score
                # sum of the weights (similarity scores)
                totalSim.setdefault(other_item,0.0)
                totalSim[other_item]+=sim_score
    # calculate weighted average for each unrated movie
    rankings=[(score/totalSim[item],item) for item,score in scores.items()]
    # rank highest-->lowest
    rankings.sort()
    rankings.reverse()
    return rankings

if __name__ == "__main__":

    data = {'movieId':[1,2,2],'userId':[1,1,2],'rating':[4.0,4.5,4.0]}
    df=pd.DataFrame.from_dict(data,orient='index')
    df_newmov=df.T
    print df_newmov.head()

    import sys
    user_name = raw_input("Enter user id: ")
    while (user_name not in str(df_newmov['userId'].unique())) :
        print 'Cannot locate username. Please retry.'
        user_name = raw_input("Enter username: ")


    ## dictionary with movie item as key and similarity scores
    ## with other movie items as values
    itemScores=calculate_similar_items(df_newmov,n=5)
    print itemScores

    rankings=get_recommendations(df_newmov,itemScores,user_name)
'''    if(len(rankings)==0):
        print 'You have seen all the movies. No movies to recommend.'
    else:
        print
        print 'Recommended movies for user id ' + str(user_name) + ':'
        expected_score, recs = zip(*rankings)
        for score,rec in rankings:
            if(score >= 3.5):
                print '=============================================='
                print 'Movie ID: ' + str(rec)
                print 'Movie: ' + str(df_newmov[['title']][df_newmov['movieId']==rec][:1])
                print 'Expected rating by user (maximum = 5 stars): ' + str(score)
                print '=============================================='
'''
