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

    n = len(index)
    if n==0:return 0

    sum_of_squares=0

    # euclidiean distance of all movies: (x_1-y_1)^2 + (x_2-y_2)^2 + ...
    sum_of_squares = np.sum(pow(x_1-x_2,2))

    # return similarity score
    return 1.0/(1.0+sum_of_squares)

# if items are rated similarly by a user,
# the items are treated as similar
def top_matches(df,item,n=5,similarity=sim_pearson):
    # list of tuples
    scores=[]

    for i, item_comb in enumerate(combinations(df.columns,2)):
        if(item in item_comb ):
            df_pair = df[[item_comb[0],item_comb[1]]].dropna()
            if(item_comb[0]!=item) : other_item=item_comb[0]
            else : other_item=item_comb[1]

            # store similarity score
            scores.append((similarity(df_pair.index,df_pair[item_comb[0]],df_pair[item_comb[1]]),other_item))

    scores.sort()
    scores.reverse()
    # return top n scores
    return scores[0:n]

def calculate_similar_items(df,n=10):
    # Create dictionary of items that stores other similarly rated items
    result={}

    # iterate over items
    for item in df.columns:
        # calculate similarity scores between this item and others
        # and store as a tuple
        result[item] = top_matches(df,item,n=n,similarity=sim_distance)

    return result

def get_recommendations(df,itemScores,usr):
    # list of tuples
    rankings=[]
    totalSim={}
    scores={}

    df_t = df.T
    df_usr = df_t[usr].dropna()

    # Loop over items rated by this user
    for item in df_usr.index:
        # Loop over items similar to this item
        for (sim_score,other_item) in itemScores[item]:
            # ignore item if user has alread rated it
            if other_item in df_usr: continue
            # unrated item is scored by weighting the rating of this item
            # with the similarity score between this item and the current unrated item
            rating=df_usr[item]
            scores.setdefault(other_item,0.0)
            # weighted sum for other_time
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
    movie_user_preferences={'Jill': {'Avenger: Age of Ultron': 7.0,
                            'Django Unchained': 6.5,
                            'Gone Girl': 9.0,
                            'Kill the Messenger': 8.0},
                            'Julia': {'Avenger: Age of Ultron': 10.0,
                                'Django Unchained': 6.0,
                                'Gone Girl': 6.5,
                                'Kill the Messenger': 6.0,
                                'Zoolander': 6.5},
                            'Max': {'Avenger: Age of Ultron': 7.0,
                                'Django Unchained': 7.0,
                                'Gone Girl': 10.0,
                                'Horrible Bosses 2': 6.0,
                                'Kill the Messenger': 5.0,
                                'Zoolander': 10.0},
                            'Robert': {'Avenger: Age of Ultron': 8.0,
                                'Django Unchained': 7.0,
                                'Horrible Bosses 2': 5.0,
                                'Kill the Messenger': 9.0,
                                'Zoolander': 9.0},
                            'Sam': {'Avenger: Age of Ultron': 10.0,
                                'Django Unchained': 7.5,
                                'Gone Girl': 6.0,
                                'Horrible Bosses 2': 3.0,
                                'Kill the Messenger': 5.5,
                                'Zoolander': 7.0},
                            'Toby': {'Avenger: Age of Ultron': 8.5,
                                'Django Unchained': 9.0,
                                'Zoolander': 2.0},
                            'William': {'Avenger: Age of Ultron': 6.0,
                                'Django Unchained': 8.0,
                                'Gone Girl': 7.0,
                                'Horrible Bosses 2': 4.0,
                                'Kill the Messenger': 6.5,
                                'Zoolander': 4.0}}



    # create df with username as index
    df=pd.DataFrame.from_dict(movie_user_preferences, orient='index')

    print df
    df_t = df.T
    print df_t

    import sys
    #print top scores for given user
    null_values = 0
    user_name = raw_input("Enter username: ")
    if (user_name not in df_t.columns) :
        print 'Cannot locate username.'
        ans = raw_input("Would you like to sign up [Y/n] ")
        if(ans=='Y') :
            new_user=raw_input('Please choose a username. ')
            df_t[new_user] = np.nan
            print df_t
            ans = raw_input("Would you like to rate movies [Y/n] ")
            if(ans=='Y'):
                print 'Rate on scale from 1.0-10.0.'
                for i,movie in enumerate(df_t.index):
                    try:
                        rating = raw_input(str(movie)+ ': ')
                        df_t[new_user][i]=rating
                    except:
                        pass
                null_values = df_t[new_user].isnull().sum()
            else : sys.exit()
        else: sys.exit()

    df=df_t.T
    print df

    if (null_values!=len(df)):

        itemScores=calculate_similar_items(df,n=10)
        #user_name = 'Toby'
        rankings=get_recommendations(df,itemScores,user_name)
        print
        print 'Recommendations for ' + str(user_name)
        expected_score, recs = zip(*rankings)
        for score,rec in rankings:
            print 'Movie: ' + str(rec) + ', Expected rating by user: ' + str(score)
    else:
        print 'Please rate movies in order to receive preferences.'
        sys.exit()

