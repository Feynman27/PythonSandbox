#!/usr/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas.tools.rplot as rplot

do_plot_all = True
# evenly-spaced values in increments of 0.1
p1 = np.arange(0.0,30.0,0.1)

# rows,cols,plot number
fig1 =plt.figure(1)
ax=fig1.add_subplot(211)

plt.plot(p1,np.sin(p1)/p1,'b--')

ax=fig1.add_subplot(212)
plt.plot(p1,np.cos(p1),'r--')
if(do_plot_all==False): plt.close();

# Playing with text
# sample 5 #'s from uniform distribution
n = np.random.random_sample((5,))

# y-vals from array
fig2 = plt.figure(2)
ax = fig2.add_subplot(111)
rect=ax.bar(np.arange(len(n)),n)
ax.set_xlabel('Indices')
ax.set_ylabel('Value Sampled from Continuous Uniform Distribution')
# add text to the figure
plt.text(1, .7, r'$\mu=' + str(np.round(np.mean(n), 2)) + ' $')
if(do_plot_all==False): plt.close();

# Annotate certain section of plot
fig3 = plt.figure(3)
ax = fig3.add_subplot(111)
t = np.arange(0.0,5.0,0.01)
s = np.cos(2*np.pi*t)
line,=plt.plot(t,s,lw=2)
plt.annotate('local max',xy=(2,1),xytext=(3,1.5),arrowprops=dict(facecolor='black',shrink=0.05))
plt.ylim(-2,2)
if(do_plot_all==False): plt.close();

# Styling plots
fig4 = plt.figure(4)
plt.style.use('ggplot')
ax=fig4.add_subplot(111)
plt.plot([1,2,3,4],[1,4,9,16])
if(do_plot_all==False): plt.close();

fig5 = plt.figure(5)
plt.style.use('fivethirtyeight')
ax=fig5.add_subplot(111)
plt.plot([1,2,3,4],[1,4,9,16])
if(do_plot_all==False): plt.close();

fig6=plt.figure(6)
with plt.style.context(('dark_background')):
    plt.plot([1,2,3,4],[1,4,9,16])
ax=fig6.add_subplot(111)
if(do_plot_all==False): plt.close();

# box plot
'''
Q3: This is the 75th percentile value of the data. It's also called the upper hinge.
Q1: This is the 25th percentile value of the data. It's also called the lower hinge.
Box: This is also called a step. It's the difference between the upper hinge and the lower hinge.
Median: This is the midpoint of the data.
Max: This is the upper inner fence. It is 1.5 times the step above Q3.
Min: This is the lower inner fence. It is 1.5 times the step below Q1.
'''
np.random.seed(10)
box_data1 = np.random.normal(100,10,200)
box_data2 = np.random.normal(80,30,200)
box_data3 = np.random.normal(90,20,200)

# combine data to a list
data_to_plot=[box_data1,box_data2,box_data3]
# Create the boxplot
fig7 = plt.figure(7)
ax=fig7.add_subplot(111)
bp = plt.boxplot(data_to_plot,patch_artist=True)
#outline,fill color and linewidth of boxes
for box in bp['boxes']:
    # outline color
    box.set(color='#7570b3', linewidth=2)
    # fill color
    box.set( facecolor = '#1b9e77')

## change color and linewidth of the whiskers
for whisker in bp['whiskers']:
    whisker.set(color='#7570b3', linewidth=2)

## change color and linewidth of the caps
for cap in bp['caps']:
    cap.set(color='#7570b3', linewidth=2)

## change color and linewidth of the medians
for median in bp['medians']:
    median.set(color='#b2df8a', linewidth=2)

## change the style of fliers and their fill
for flier in bp['fliers']:
    flier.set(marker='o', color='#e7298a', alpha=0.5)
if(do_plot_all==False): plt.close();

## heatmaps

# 10x6 matrix propagated with
# random samples from a uniform distribution.
data=np.random.rand(10,6)
rows=list('ZYXWVUTSRQ') # y label
columns=list('ABCDEF') # x label

fig8 = plt.figure(8)
ax=fig8.add_subplot(111)
# add color to elements
plt.pcolor(data)

# uncomment to change default colors
plt.pcolor(data,cmap=plt.cm.Reds,edgecolors='k')
# add row/column labels
plt.yticks(np.arange(0,10)+0.5,rows)
plt.xticks(np.arange(0,6)+0.5,columns)

# heat maps with binned data

# generate test data in x and y
# from standarn normal distribution
x=np.random.randn(8873)
y=np.random.randn(8873)

# bin data
heatmap,xedges,yedges=np.histogram2d(x,y,bins=50)
extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]]
if(do_plot_all==False): plt.close();

fig9 = plt.figure(9)
ax=fig9.add_subplot(111)
plt.imshow(heatmap,extent=extent)
if(do_plot_all==False): plt.close();

## scatter plots and projections
fig10 = plt.figure(10,figsize=(8,8))
from matplotlib.ticker import NullFormatter
x = np.random.randn(1000)
y = np.random.randn(1000)

# don't want x/y labels
nullfmt   = NullFormatter() # no labels

# size, height, and width of the scatter and projections
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
bottom_h = left_h = left+width+0.02

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.2, height]

# plot axes for scatter and x,y projections
#ax=fig10.add_subplot(111)
axScatter = plt.axes(rect_scatter)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)

# eliminate x,y labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)
# plot scatter
axScatter.scatter(x, y)

# now determine nice x,y limits by hand
binwidth = 0.25
xymax = np.max( [np.max(np.fabs(x)),np.max(np.fabs(y))] )
lim = (int(xymax/binwidth)+1)*binwidth
axScatter.set_xlim( (-lim, lim) )
axScatter.set_ylim( (-lim, lim) )

# list of interval values
bins = np.arange(-lim, lim + binwidth, binwidth)

axHistx.hist(x, bins=bins)
axHisty.hist(y, bins=bins, orientation='horizontal')

axHistx.set_xlim( axScatter.get_xlim() )
axHisty.set_ylim( axScatter.get_ylim() )
if(do_plot_all==False): plt.close();

## scatter matrix
# dataframe with four columns of standard gaussian RVs
df = pd.DataFrame(np.random.randn(1000,4),columns=['a','b','c','d'])
#fig11 = plt.figure(11)
#ax=fig11.add_subplot(111)
spm = pd.tools.plotting.scatter_matrix(df,alpha=0.2,figsize=(8,8),diagonal='hist')
if(do_plot_all==False): plt.close();
#fig12 = plt.figure(12)
#ax=fig12.add_subplot(111)
spm = pd.tools.plotting.scatter_matrix(df,alpha=0.2,figsize=(8,8),diagonal='kde')
if(do_plot_all==False): plt.close();

# stacked histograms
# generate 10x4 matrix where each column is a separate distribution
df = pd.DataFrame(np.random.rand(10,4),columns=['p', 'q', 'r', 's'])
df.plot(kind='area')
if(do_plot_all==False): plt.close();

# draw unstacked
df.plot(kind='area', stacked=False)
if(do_plot_all==False): plt.close();

# bubble charts: scatter plot with an additional dimension,
# where the size of the bubble implies larger value
plt.style.use('ggplot')
# a,b,c are RVs
df = pd.DataFrame(np.random.rand(50, 3), columns=['a', 'b', 'c'])
# RV c is designated as the size variable
df.plot(kind='scatter', x='a', y='b', s=df['c']*400)
if(do_plot_all==False): plt.close();

## hexagon bin plot
## Used when scatter plot is too dense.
## Color intensity used to interpret concentration
## of points.

df = pd.DataFrame(np.random.randn(1000, 2), columns=['a', 'b'])
df['b'] = df['b'] + np.arange(1000)
df.plot(kind='hexbin', x='a', y='b', gridsize=25)
if(do_plot_all==False): plt.close();

## Trellis plots
## Grid layout used for categorization
## Plot tips as a function of bill based
## on sex and smoker/non-smoker
tips_data=pd.read_csv('../data/tips.csv')
fig17=plt.figure(17)
#ax=fig17.add_subplot(111)
plot = rplot.RPlot(tips_data, x='total_bill', y='tip')
#plt.xlabel('total bill[$]')
#plt.ylabel('tip[$]')
# sex=row, smoker=column
plot.add(rplot.TrellisGrid(['sex', 'smoker']))
plot.add(rplot.GeomHistogram())
plot.render(plt.gcf())
if(do_plot_all==False): plt.close();

#import seaborn as sns
#g = sns.FacetGrid(tips_data, row="sex", col="smoker")
#g.map(plt.hist, "total_bill")

# KDE instead of histogrm
fig18=plt.figure(18)
plot = rplot.RPlot(tips_data, x='total_bill', y='tip')
plot.add(rplot.TrellisGrid(['sex', 'smoker']))
plot.add(rplot.GeomDensity())
plot.render(plt.gcf())
if(do_plot_all==False): plt.close();

# scatter
fig19=plt.figure(19)
plot = rplot.RPlot(tips_data, x='total_bill', y='tip')
plot.add(rplot.TrellisGrid(['sex', 'smoker']))
plot.add(rplot.GeomScatter())
plot.add(rplot.GeomPolyFit(degree=2))
plot.render(plt.gcf())
if(do_plot_all==False): plt.close();

# 2D KDE
fig20=plt.figure(20)
plot = rplot.RPlot(tips_data, x='total_bill', y='tip')
plot.add(rplot.TrellisGrid(['sex', 'smoker']))
plot.add(rplot.GeomScatter())
plot.add(rplot.GeomDensity2D())
plot.render(plt.gcf())
if(do_plot_all==False): plt.close();

# 3D plotting of surfaces
from mpl_toolkits.mplot3d import Axes3D
fig21=plt.figure(21)
ax=Axes3D(fig21)
X=np.arange(-4,4,0.25)
Y=np.arange(-4,4,0.25)
X,Y=np.meshgrid(X,Y)
R=np.sqrt(X**2+Y**2)
Z=np.sin(R)
# uncomment to adjust the view
#ax.view_init(elev=0., azim=0)
#ax.view_init(elev=50., azim=0)
ax.view_init(elev=50., azim=30)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')
if(do_plot_all==False): plt.close();

## plot all figures
plt.show()
