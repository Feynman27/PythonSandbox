#!/usr/bin/python

'''
Build user-based collaborative filter by finding
users who are similar to each other and
build a similarity score '''

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

def plot_figs(usrs,x,y,xlbl,ylbl):
    plt.scatter(x,y)
    plt.xlabel('Rating for ' + str(xlbl))
    plt.ylabel('Rating for ' + str(ylbl))
    # uncomment for showing username on figure
    for i,usr in enumerate(usrs):
        plt.annotate(usr,(x[i],y[i]) )
    return 0

def sim_distance(df,usr1,usr2):
    sum_of_squares=0
    for movie in df.columns[:]:
        #print 'Movie ratings for: ' + str(movie)
        #print 'User 1: ' + str(usr1) + ', ' + str(df[movie][usr1])
        #print 'User 2: ' + str(usr2) + ', ' + str(df[movie][usr2])
        # If one or more values is null, go to next movie column
        if( (np.isnan(df[movie][usr1]) ) | np.isnan(df[movie][usr2]) ) : continue

        # euclidiean distance of all movies: (x11-x21)^2 + (x12-x22)^2 + ...
        sum_of_squares += np.sum([pow(df[movie][usr1]-df[movie][usr2],2)])

    # return similarity score
    return 1.0/(1.0+sum_of_squares)

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

    # create list
    '''
    data=[]
    for i,key in enumerate(movie_user_preferences.keys() ):
        try:
            data.append((key
                        ,movie_user_preferences[key]['Gone Girl'] if 'Gone Girl' in movie_user_preferences[key] else 'NaN'
                        ,movie_user_preferences[key]['Horrible Bosses 2'] if 'Horrible Bosses 2' in movie_user_preferences[key] else 'NaN'
                        ,movie_user_preferences[key]['Django Unchained'] if 'Django Unchained' in movie_user_preferences[key] else 'NaN'
                        ,movie_user_preferences[key]['Zoolander'] if 'Zoolander' in movie_user_preferences[key] else 'NaN'
                        ,movie_user_preferences[key]['Avenger: Age of Ultron'] if 'Avenger: Age of Ultron' in movie_user_preferences[key] else 'NaN'
                        ,movie_user_preferences[key]['Kill the Messenger'] if 'Kill the Messenger' in movie_user_preferences[key] else 'NaN' ))
            # iterate over movie ratings (values of the nested dict)
        # if no entry, skip
        except:
            pass
    '''
    # create df
    #df=pd.DataFrame(data=data,columns=['user','Gone_Girl','Horrible_Bosses_2','Django_Unchained','Zoolander','Avenger_Age_of_Ultron','Kill_the_Messenger'])
    # keys are rows
    #df=pd.DataFrame.from_dict(movie_user_preferences, orient='index').reset_index()
    df=pd.DataFrame.from_dict(movie_user_preferences, orient='index')
    # label columns
    df.columns=['Gone_Girl','Horrible_Bosses_2','Django_Unchained','Zoolander','Avenger_Age_of_Ultron','Kill_the_Messenger']
    print df

    #import scipy.special as spec
    #fig,ax=plt.subplots(3,int(spec.binom(len(df.columns),2)/2))
    for i,movie_combo in enumerate(combinations(df.columns,2)):
        fig = plt.figure(i)
        plot_figs(df.index,df[movie_combo[0]],df[movie_combo[1]],movie_combo[0],movie_combo[1])

    # iterate through all user combinations
    for c in combinations(df.index,2):
        #print 'User 1: ' + str(c[0]) + ', User 2: ' + str(c[1])
        # compute euclidean distance between points
        # sqrt(pow(x2-x1,2)+pow(y2-y2,2))
        similarity_score=sim_distance(df,c[0],c[1])
        if(similarity_score>0.2):
            print c[0] + ' and ' + c[1] + ' have similar movie preferences.'

    plt.show()
