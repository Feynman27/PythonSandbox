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
import sys
from sklearn import linear_model
import matplotlib.gridspec as gridspec
from scipy.stats.stats import pearsonr

numplots=1
pastItem = []
fig = plt.figure(figsize=(12,10))
plt.suptitle("Movie Rating Correlations",y=1.0)

gs1 = gridspec.GridSpec(2,3)

## Scatter plot of ratings and fit to linear model
def plot_figs(x,y,title_x,title_y):
    ax = fig.add_subplot(gs1[numplots-1])
    ax.scatter(y,x)
    lm = linear_model.LinearRegression()
    lm.fit(y.reshape((len(y.index),1)),x.reshape(len(x.index),1))
    ax.plot(y,lm.predict(y.reshape((len(y.index),1))),color='red')
    ax.set_xlabel(title_x)
    ax.set_ylabel(title_y)
    print 'Coefficient value: ' + str(lm.coef_) + '\n'
    return 0

## Similarity score based on Pearson r
def sim_pearson(index,x_1,x_2):

    ## index = list of usernames
    ## x_i = vector of reviews for movie i
    n = len(index)
    if n==0:
        return 0

    x_1 = x_1.as_matrix()
    x_2 = x_2.as_matrix()
    r,p_value = pearsonr(x_1,x_2)
    if np.isnan(r):
        r = 0
    return r

    ## Uncomment for cross-check
    '''
    ## sum of ratings for each user
    sum1=np.sum(x_1)
    sum2=np.sum(x_2)

    ## sum of squares x_i*x_i
    sumsq1 = np.sum(pow(x_1,2))
    sumsq2 = np.sum(pow(x_2,2))

    ## sum of off-diagonal products x_i*y_i
    sumsq12 = np.sum(x_1*x_2)
    '''
    ## calculate pearson correlation coeff
    '''
        E[XY] - E[X]E[Y]
        ---------------
        sqrt(E[X^2]-(E[X])^2)*sqrt(E[Y^2]-(E[Y])^2)

        which for a sample may be written as

            {n\sum x_i y_i - \sum x_i \sum y_i}
        r = ----------------
            \sqrt{n \sum x_i^2-(\sum x_i)^2} \sqrt{n \sum y_i^2-(\sum y_i)^2}
    '''
    '''
    num = sumsq12-(sum1*sum2/n)
    den =  np.sqrt((sumsq1-(pow(sum1,2)/n) )*(sumsq2-(pow(sum2,2)/n) ) )
    r=num/den
    if np.isnan(r):
        #print '[sim_pearson] WARNING: correlation coefficient is undefined. Will return r = -999'
        r = -999.0
    return r
    '''

## Sim score based on Euclidean distances
def sim_distance(index,x_1,x_2):

    ## index = list of usernames
    ## x_i = vector of reviews for movie i
    n = len(index)
    if n==0:return 0

    sum_of_squares=0

    ## euclidiean distance of all movies: (x_1-y_1)^2 + (x_2-y_2)^2 + ...
    ## from vector of ratings from all users
    sum_of_squares = np.sum(pow(x_1-x_2,2))

    ## return similarity score
    return 1.0/(1.0+np.sqrt(sum_of_squares))

## Compute similarity scores between each movie pair
## and save top scores in a dictionary.
## Key = movieId, Value = (sim_scores,other_movieIds)
def top_matches(df,item,n=5,similarity=sim_pearson,max_iter=10):
    global numplots
    global pastItem
    ## list of tuples
    scores=[]

    count = 0
    ## iterate over all movie pairs
    for i, item_comb in enumerate(combinations(df['movieId'].unique(),2)):
        ## use only those pairs with the current movie of interest
        if(item in item_comb ):
            count += 1
            ## maximum sim scores computed per movie
            if(count==max_iter) :
                count = 0
                break
            ## Drop NaN from these two columns
            if(item_comb[0]!=item) : other_item=item_comb[0]
            else : other_item=item_comb[1]

            df_item = df[['userId','rating','title']][df['movieId']==item].reset_index(drop=True)
            df_otheritem = df[['userId','rating','title']][df['movieId']==other_item].reset_index(drop=True)
            df_pair = pd.merge(df_item,df_otheritem,how='outer',left_on='userId',right_on='userId')
            df_pair = df_pair.dropna()
            df_pair=df_pair.reset_index(drop=True)

            ## store similarity score
            scores.append((similarity(df_pair['userId'],df_pair['rating_x'],df_pair['rating_y']),other_item))

            ## make plots with at least 6 observations
            if ( (len(df_pair)>5)  & (numplots<7) & (item not in pastItem) & (other_item not in pastItem)) :
                print '============================================================='
                print df_pair
                print
                plot_figs(df_pair['rating_x'],df_pair['rating_y'],str(df_pair['title_x'][0]),str(df_pair['title_y'][0]))

                numplots+=1
                pastItem.append(item)
                pastItem.append(other_item)

    ## Sort similarity scores in descending order (i.e. most similar items first)
    scores.sort()
    scores.reverse()
    ## return top n scores
    return scores[0:n]

def calculate_similar_items(df,n=10):
    ## Create dictionary that stores similarity scores
    ## between movie pairs
    result={}
    print 'Computing similarity scores. This may take a minute...'
    ## iterate over each movie item
    for i,item in enumerate(df['movieId'].unique()):
        ## limit number of movies
        if(i==200):break
        ## calculate similarity scores between this movie item and other
        ## movie items and store as a tuple

        ## Choose how to calculate similarity score (Euclidean or Pearson correlation)
        #result[item] = top_matches(df,item,n=n,similarity=sim_distance,max_iter=50)
        result[item] = top_matches(df,item,n=n,similarity=sim_pearson, max_iter=50)

    print 'Done.'
    return result

## Return recommended movies for this user
## based on predicted rating of unseen movies
def get_recommendations(df,itemScores,usr):
    ## list of tuples
    rankings=[]
    totalSim={}
    scores={}

    ## map for plotting heatmap
    dict_sim_scores = {}

    ## Ratings for only this user
    df_usr = df[['rating','userId','movieId','title']][df['userId']==int(usr)]
    df_usr = df_usr.dropna()
    df_usr = df_usr.reset_index(drop=True)
    #print 'Movie id for viewed items:'
    #print df_usr['movieId'].unique()
    #print

    ## Loop over movie items rated by this user
    for index,item in enumerate(df_usr['movieId'].unique()):
        ## make sure key exists
        if(item in itemScores.keys()):
            ## Loop over items similar to this item
            for (sim_score,other_item) in itemScores[item]:
                ## Ignore item if user has already rated it
                if (other_item in df_usr['movieId'].unique()):continue

                '''
                Unrated item is scored by a weighted average, which is calculated
                from the movie ratings of movies the user has viewed {R_i} and
                the similarity scores between the viewed (i) and unviewed (j) movies {S_ij}

                             sum_i R_i*S_ij
                R_j(pred) =  -------------
                              sum_i S_ij
                '''

                ## retrieve rating for movie user has viewed
                rating=df_usr['rating'][index]
                ## If key=other_item cannot be found, set default to 0
                scores.setdefault(other_item,0.0)
                ## weighted sum for unrated item
                scores[other_item]+=rating * sim_score
                ## sum of the weights (similarity scores)
                totalSim.setdefault(other_item,0.0)
                totalSim[other_item]+=sim_score

                ## construct mapping of viewed and unviewed
                ## movie items to similarity scores for this user
                row_indexer_item = df_usr[['title']][df_usr['movieId']==item].index[0]
                row_indexer_other_item = df[['title']][df['movieId']==other_item].index[0]
                ## title of movie user has seen
                item_title = str(df_usr.loc[row_indexer_item,'title'])
                ## title of movie user has not seen
                other_item_title = str(df.loc[row_indexer_other_item,'title'])
                dict_sim_scores[(item_title,other_item_title)] = float(sim_score)

    try:
        ## create a heatmap
        sim_scores = pd.Series(list(dict_sim_scores.values()),index=pd.MultiIndex.from_tuples(dict_sim_scores.keys()))
        df_sim_scores = sim_scores.unstack().fillna(0.)
        import seaborn as sns
        fig_heatmap = plt.figure('Similarity Scores',figsize=(10,10))
        plt.title('Top Similarity Scores of Viewed vs. Unviewed Movies for User ' + str(usr))
        sns.heatmap(df_sim_scores,linewidths=0.5)
        plt.xticks(rotation=40,ha='right',fontsize='x-small')
        plt.yticks(fontsize='x-small')
        fig_heatmap.tight_layout()

    except TypeError:
        print 'WARNING: Data sparsity. Unable to create heatmap of similarity scores.'
        pass

    ## calculate weighted average for each unrated movie
    rankings=[(score/totalSim[item],item) for item,score in scores.items()]
    ## rank highest-->lowest
    rankings.sort()
    rankings.reverse()
    return rankings

if __name__ == "__main__":

    import time
    start = time.clock()
    ## read in dataframes
    df_movies = pd.read_csv('../../data/ml-latest-small/movies.csv')
    df_ratings = pd.read_csv('../../data/ml-latest-small/ratings.csv',usecols=range(3))
    df_imdb = pd.read_csv('../../data/ml-latest-small/links.csv',usecols=range(2))

    ## merge by movieId
    df = pd.merge(df_movies,df_ratings,left_on='movieId',right_on='movieId',how='outer')
    df = pd.merge(df,df_imdb,left_on='movieId',right_on='movieId',how='outer')

    df_2 = pd.DataFrame(df.title.str.split('(',1).tolist(),columns=['title','relDate'])
    df_2['relDate'] = df_2['relDate'].apply(lambda x: str(x).replace(')', ''))
    df = pd.merge(df,df_2,how='outer',left_index=True,right_index=True)
    df = df.drop('title_x',1)
    df.rename(columns={'title_y':'title'},inplace=True)
    df['relDate'] = df['relDate'].convert_objects(convert_numeric=True)

    ## Use only movie with release date after 2010
    df_newmov = df[df['relDate']>2010]
    df_newmov = df_newmov.reset_index(drop=True)
    print df_newmov.head()
    #print df_newmov.tail()

    user_name = raw_input("Enter user id: ")
    if(user_name == 'quit'): sys.exit()
    while (user_name not in str(df_newmov['userId'].unique())) :
        print 'Cannot locate username. Please retry.'
        user_name = raw_input("Enter username: ")
        if(user_name == 'quit'): sys.exit()


    ## dictionary with movie item as key and similarity scores
    ## with other movie items as values
    itemScores=calculate_similar_items(df_newmov,n=5)

    rankings=get_recommendations(df_newmov,itemScores,user_name)
    if(len(rankings)==0):
        print 'No movies to recommend.'
    else:
        print
        print 'Recommended movies for user id ' + str(user_name) + ':'
        expected_score, recs = zip(*rankings)
        liked = 0
        summary_table = []
        top_movies = []
        pred_ratings = []

        for score,rec in rankings:
            if(score >= 3.0):
                liked += 1
                ## find row_indexer
                row_indexer = df_newmov[['title']][df_newmov['movieId']==rec].index[0]
                movie = str(df_newmov.loc[row_indexer,'title'])
                score = '%2.1f' % score
                print '=============================================='
                print str(liked)+'.'
                print 'Movie ID: ' + str(rec)
                print 'Movie: ' + movie
                print 'Expected rating by user (maximum = 5 stars): ' + str(score)
                print

                top_movies.append(movie)
                pred_ratings.append(float(score))

        if(liked == 0) :
            print 'Unable to recommend movies for this user.'
            sys.exit()

        try:
            fig2 = plt.figure('Predicted Ratings',figsize=(8,10))
            plt.title('Predicted Ratings for User ' + str(user_name))
            ax = fig2.add_subplot(111)
            rect = ax.bar(np.arange(len(pred_ratings)),pred_ratings,color='g',alpha=0.5)
            ax.set_xticks(np.arange(len(top_movies)))
            ax.set_xticklabels(top_movies,rotation=40,fontsize='x-small',ha='right')
            ax.set_ylabel('Predicted User Rating')
            ax.set_ylim((0.0,5.5))
            ax.set_xlim(0,(np.arange(len(top_movies))[-1])+1)
            ax.grid(True)
            plt.axhline(y=5.0,color='r',linestyle='dotted',lw=5)
            fig2.tight_layout()
        except TypeError:
            print 'WARNING: Data sparsity. Unable to determine predicted ratings.'
            pass

    print 'Elapsed time: ' + str((time.clock()-start)/60.0) + ' minutes.'
    gs1.tight_layout(fig)
    plt.show()
