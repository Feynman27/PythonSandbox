#!/usr/bin/python

'''
UN data on educational level and GDP
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plotter(df,x,y,d_color):
    for c in d_color.keys():
        color=d_color[c]
        df[df['countrySegment']==c].plot(kind='scatter',x=x,y=y,label=c,ax=ax,color=color)

if __name__ == "__main__":
    df=pd.read_csv('../data/UN.csv')
    print('-----')
# check types
    print [(col,type(df[col][0])) for col in df.columns]
# check fill rate of columns by calculating
# fraction of rows with non-null values in a given column
    print('Percentage of the values complete in the columns')
    s_col_fill = df.count(0)/df.shape[0]*100.0
    print s_col_fill

# select features with high fill rate
    df = df[['lifeMale','lifeFemale','infantMortality','GDPperCapita']]
    df=df.dropna()

# Determine number of country clusters
    from math import sqrt
    from scipy.stats import pearsonr
    from sklearn.cluster import KMeans
    from scipy.cluster.vq import kmeans,vq
    from scipy.spatial.distance import cdist
    K=range(1,10)
# perform k-means on set up to 10 clusters
# centroids are adjusted until distortion is below a certain threshold
    k_clusters=[kmeans(df.values,k) for k in K]
    print k_clusters[:3]

# distance between observations and centroid of each cluster using 1-10 clusters
    euclidean_centroid = [cdist(df.values, centroid, 'euclidean') for (centroid,var) in k_clusters]

    print '-----with 1 cluster------'
    print euclidean_centroid[0][:5]
    print '-----with 2 clusters------'
    print euclidean_centroid[1][:5]

# find min distance of each observed point to closest centroid
    dist = [np.min(D,axis=1) for D in euclidean_centroid]

    print '-----convergence with 1 cluster------'
    print dist[0][:5]
    print '-----convergence with 2 clusters------'
    print dist[1][:5]

# average distance to centroid (up to 10 clusters)
    avgDistance = [sum(d)/df.values.shape[0] for d in dist]

# plot elbow curve to find optimum number of clusters
# i.e. region where distance-to-centroid starts to flatten out
    ifig=1
    kIdx = 2
    fig = plt.figure(ifig); ifig+=1
    ax=fig.add_subplot(111)
    ax.plot(K,avgDistance,'b*-')
    ax.plot(K[kIdx],avgDistance[kIdx],marker='o',markersize=12.0,markeredgewidth=2,markeredgecolor='r',markerfacecolor='None')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Distance to closest cluster centroid')
    plt.title('K-Means Elbow Curve')

# Apply k-means algo to cluster countries into k=3 segments: underdeveloped, developing, developed

    km = KMeans(3,init='k-means++',random_state=3425)
    km.fit(df.values)
# segment into 3 groups
    df['countrySegment']=km.predict(df.values)
    print df[:5]
# Show segments based on mean GDP per capita
    df.groupby('countrySegment').GDPperCapita.mean()

# shows 0=developing, 1=underdeveloped, 2=developed
    clust_map={0:'Developing',1:'UnderDeveloped',2:'Developed'}
    df['countrySegment'] = df['countrySegment'].map(clust_map)
    d_color={'Developing':'b','UnderDeveloped':'r','Developed':'g'}
    print df[:10]

    fig,ax = plt.subplots() ;ifig+=1
    plotter(df,'GDPperCapita','infantMortality',d_color)
    handles,labels=ax.get_legend_handles_labels()
    _ = ax.legend(handles,labels,loc='upper right')

    fig,ax = plt.subplots() ;ifig+=1
    plotter(df,'GDPperCapita','lifeMale',d_color)
    handles,labels=ax.get_legend_handles_labels()
    _ = ax.legend(handles,labels,loc='lower right')

    fig,ax = plt.subplots() ;ifig+=1
    plotter(df,'GDPperCapita','lifeFemale',d_color)
    handles,labels=ax.get_legend_handles_labels()
    _ = ax.legend(handles,labels,loc='lower right')

    plt.show()
