#!/usr/bin/python

'''
Build user-based collaborative filter by finding
users who are similar to each other and
build a similarity score from the Pearson correlation'''

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import scipy.stats as stats

def sim_distance(movies,x_1,x_2):

    n = len(movies)
    if n==0:return 0

    sum_of_squares=0

    # euclidiean distance of all movies: (x_1-y_1)^2 + (x_2-y_2)^2 + ...
    sum_of_squares = np.sum(pow(x_1-x_2,2))

    # return similarity score
    return 1.0/(1.0+sum_of_squares)

def sim_pearson(movies,x_1,x_2):
    n = len(movies)
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

def top_matches(df,usr,n=5,similarity=sim_pearson):
    # list of tuples
    scores=[]

    for i, usr_comb in enumerate(combinations(df.columns,2)):
        if(usr in usr_comb ):
            df_pair = df[[usr_comb[0],usr_comb[1]]].dropna()
            if(usr_comb[0]!=usr) : other_usr=usr_comb[0]
            else : other_usr=usr_comb[1]

            # store similarity score
            scores.append((similarity(df_pair.index,df_pair[usr_comb[0]],df_pair[usr_comb[1]]),other_usr))

    scores.sort()
    scores.reverse()
    # return top n scores
    return scores[0:n]

def get_recommendations(df,usr,similarity=sim_pearson):
    # list of tuples
    rankings={}

    totals={}
    simSums={}
    for i, usr_comb in enumerate(combinations(df.columns,2)):
        if(usr in usr_comb):

            df_pair = df[[usr_comb[0],usr_comb[1]]].dropna()
            sim =similarity(df_pair.index,df_pair[usr_comb[0]],df_pair[usr_comb[1]])
            # ignore low scores
            if(sim<=0.0) : continue
            # only score movies not seen by user
            # but seen by other users

            if(usr_comb[0]!=usr) : other_usr=usr_comb[0]
            else : other_usr=usr_comb[1]
            for i,movie in enumerate(df.index):
                if ( np.isnan(df[usr][i]) and not np.isnan(df[other_usr][i]) ):

                    totals.setdefault(movie,0)
                    totals[movie]+=(sim*df[other_usr][i])
                    # sum of weights
                    simSums.setdefault(movie,0)
                    simSums[movie]+=sim

    rankings = [(total/simSums[movie],movie) for movie,total in totals.items()]
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


    import pandas as pd

    # create df with username as index
    df=pd.DataFrame.from_dict(movie_user_preferences, orient='index')
    # label columns
    df.columns=['Gone_Girl','Horrible_Bosses_2','Django_Unchained','Zoolander','Avenger_Age_of_Ultron','Kill_the_Messenger']
    print df

    df2 = df.T
    print df2
    # loop over all user pairs

    import sys
    #print top scores for given user
    null_values = 0
    user_name = raw_input("Enter username: ")
    if (user_name not in df2.columns) :
        print 'Cannot locate username.'
        ans = raw_input("Would you like to sign up [Y/n] ")
        if(ans=='Y') :
            new_user=raw_input('Please choose a username. ')
            df2[new_user] = np.nan
            print df2
            ans = raw_input("Would you like to rate movies [Y/n] ")
            if(ans=='Y'):
                print 'Rate on scale from 1.0-10.0.'
                for i,movie in enumerate(df2.index):
                    try:
                        rating = raw_input(str(movie)+ ': ')
                        df2[new_user][i]=rating
                    except:
                        pass
                null_values = df2[new_user].isnull().sum()
            else : sys.exit()
        else: sys.exit()

    if (null_values!=len(df2)):
        n=3
        print 'Username found. Retrieving top ' + str(n) + ' users with similar preferences for ' + str(user_name)
        print 'Pearson: ' + str(top_matches(df2,user_name,n,similarity = sim_pearson))
        print 'Euclidean: ' + str(top_matches(df2,user_name,n,similarity = sim_distance))
    else:
        print 'Please rate movies in order to receive preferences.'
        sys.exit()

    rankings=get_recommendations(df2,user_name,similarity=sim_distance)
    print rankings
    _,recs = zip(*rankings)
    print 'Movies recommended for ' + str(user_name) + ':'
    for rec in recs:
        print rec
