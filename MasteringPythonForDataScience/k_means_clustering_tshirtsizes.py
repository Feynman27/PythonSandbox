#!/usr/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('../data/tshirt_sizes.csv')
print df[:10]

d_color={'S':'b','M':'r','L':'g'}

fig,ax=plt.subplots()
for size in d_color.keys():
    color=d_color[size]
    df[df['Size']==size].plot(kind='scatter',x='Height',y='Weight',label=size,ax=ax,color=color)
handles,labels=ax.get_legend_handles_labels()
_ = ax.legend(handles,labels,loc='upper left')

# Predict T-shirt size based on person's height and weight
from math import sqrt
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans,vq
from scipy.spatial.distance import cdist

# initialize based on 3 clusters: small, medium, large
km = KMeans(3,init='k-means++',random_state=3425)
# input height and weight
km.fit(df[['Height','Weight']])

# predict which segment (ht,wt) combination belongs to
df['SizePrediction'] = km.predict(df[['Height','Weight']])

df.groupby(['Size','SizePrediction']).Size.count()
print pd.crosstab(df.Size,df.SizePrediction,rownames=['Size Truth'],colnames=['SizePrediction'])

c_map={0:'L',1:'S',2:'M'}

# Map prediction to size labels (S,M,L)
df['SizePrediction'] = df['SizePrediction'].map(c_map)
print df['SizePrediction'][:10]

# plot predicted sizes in (Ht,Wt) space
fig_pred,ax=plt.subplots()
for size in d_color.keys():
    color=d_color[size]
    df[df['SizePrediction']==size].plot(kind='scatter',x='Height',y='Weight',label=size,ax=ax,color=color)
handles,labels=ax.get_legend_handles_labels()
_ = ax.legend(handles,labels,loc='upper left')
plt.show()
