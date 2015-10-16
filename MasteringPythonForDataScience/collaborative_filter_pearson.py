#!/usr/bin/python

'''
Build user-based collaborative filter by finding
users who are similar to each other and
build a similarity score from the Pearson correlation'''

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import scipy.stats as stats

def plot_figs(movies,x,y,xlbl,ylbl):
    # check rows for null values
    plt.scatter(x,y)
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    # uncomment for showing movie name on figure
    for i,movie in enumerate(movies):
        plt.annotate(movie,(x[i],y[i]) )
    return 0

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

    for i,usr in enumerate(combinations(df2.columns,2)):
        # select only current pair columns
        df_pair = df2[[ usr[0],usr[1] ]]
        # remove movie rows with null values
        df_pair=df_pair.dropna()
        # hack
        #if (i==0) : break
        fig = plt.figure(i)
        plot_figs(df_pair.index,df_pair[usr[0]],df_pair[usr[1]],usr[0],usr[1])
        # return correlation coeff and p-value
        # null hypo is that user preferences are different
        corr,p = stats.pearsonr(df_pair[usr[0]],df_pair[usr[1]])
        if (p<0.05):
            print usr[0] + ' and ' + usr[1] + ' have similar movie preferences.'

        # uncomment for cross-check
        corr_cc = sim_pearson(df_pair.index,df_pair[usr[0]],df_pair[usr[1]])
        print 'Pearson correlation coefficient'
        print '-------------------------------'
        print 'Method 1: ' + str(corr)
        print 'Method 2: ' + str(corr_cc)
        print '-------------------------------'

    plt.show()
