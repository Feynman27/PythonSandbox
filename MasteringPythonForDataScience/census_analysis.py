#!/usr/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('../data/census.csv')

# null hypo that
# older people earn a higher income

fig = plt.figure(1)
hist_above_50 = plt.hist(data[data['greater_than_50k'] == True]['age'].values,10,facecolor='green',alpha=0.5)
plt.title('Age Distribution')
plt.xlabel('Age[yrs]')
plt.ylabel('Number of US citizens with Income > 50k [USD]')

fig = plt.figure(2)
hist_below_50 = plt.hist(data[data['greater_than_50k'] == False].age.values,10,facecolor='green',alpha=0.5)
plt.title('Age Distribution')
plt.xlabel('Age[yrs]')
plt.ylabel('Number of US citizens with Income < 50k [USD]')

# null hypo that income depends on
# working class

fig = plt.figure(3)
dist_data = pd.concat([data[data['greater_than_50k'] == True].groupby('workclass')['workclass'].count(),data[data['greater_than_50k'] == False].groupby('workclass')['workclass'].count()],axis=1)
dist_data.columns=['gt50','lt50']
dist_data_final = 100.0*dist_data['gt50']/(dist_data['gt50']+dist_data['lt50'])
dist_data_final.sort(ascending=False)

ax=dist_data_final.plot(kind='bar',color='r',y='Percentage')
ax.set_xticklabels(dist_data_final.index,rotation=30,fontsize=8,ha='right')
ax.set_xlabel('Working Class')
ax.set_ylabel('US citizens with income > $50k [%]')

# Hypo: income increase with educational level
fig = plt.figure(4)
dist_data = pd.concat([data[data['greater_than_50k'] == True].groupby('education')['education'].count(),data[data['greater_than_50k'] == False].groupby('education')['education'].count()],axis=1)
dist_data.columns=['gt50','lt50']
dist_data_final = 100.0*dist_data['gt50']/(dist_data['gt50']+dist_data['lt50'])
dist_data_final.sort(ascending=False)

ax=dist_data_final.plot(kind='bar',color='r',y='Percentage')
ax.set_xticklabels(dist_data_final.index,rotation=30,fontsize=8,ha='right')
ax.set_xlabel('Education Class')
ax.set_ylabel('US citizens with income > $50k [%]')

# Income is pos. correlated with marriage status
fig = plt.figure(5)
dist_data = pd.concat([data[data['greater_than_50k'] == True].groupby('marital_status')['marital_status'].count(),data[data['greater_than_50k'] == False].groupby('marital_status')['marital_status'].count()],axis=1)
dist_data.columns=['gt50','lt50']
dist_data_final = 100.0*dist_data['gt50']/(dist_data['gt50']+dist_data['lt50'])
dist_data_final.sort(ascending=False)

ax=dist_data_final.plot(kind='bar',color='r',y='Percentage')
ax.set_xticklabels(dist_data_final.index,rotation=30,fontsize=8,ha='right')
ax.set_xlabel('Marital Status')
ax.set_ylabel('US citizens with income > $50k [%]')

# Income depends on race
fig = plt.figure(6)
dist_data = pd.concat([data[data['greater_than_50k'] == True].groupby('race')['race'].count(),data[data['greater_than_50k'] == False].groupby('race')['race'].count()],axis=1)
dist_data.columns=['gt50','lt50']
dist_data_final = 100.0*dist_data['gt50']/(dist_data['gt50']+dist_data['lt50'])
dist_data_final.sort(ascending=False)

ax=dist_data_final.plot(kind='bar',color='r',y='Percentage')
ax.set_xticklabels(dist_data_final.index,rotation=30,fontsize=8,ha='right')
ax.set_xlabel('Racial Class')
ax.set_ylabel('US citizens with income > $50k [%]')

# Income depends on gender
fig = plt.figure(7)
dist_data = pd.concat([data[data['greater_than_50k'] == True].groupby('gender')['gender'].count(),data[data['greater_than_50k'] == False].groupby('gender')['gender'].count()],axis=1)
dist_data.columns=['gt50','lt50']
dist_data_final = 100.0*dist_data['gt50']/(dist_data['gt50']+dist_data['lt50'])
dist_data_final.sort(ascending=False)

ax=dist_data_final.plot(kind='bar',color='r',y='Percentage')
ax.set_xticklabels(dist_data_final.index,rotation=30,fontsize=8,ha='right')
ax.set_xlabel('Gender Class')
ax.set_ylabel('US citizens with income > $50k [%]')

# Income depends on hour-per-week
fig = plt.figure(8)
hist_above_50 = plt.hist(data[data['greater_than_50k']==True].hours_per_week.values,10,facecolor='green', alpha=0.5)
plt.xlabel('US citizens with income > $50K')
plt.ylabel('Hours worked per week')
fig = plt.figure(9)
hist_below_50 = plt.hist(data[data['greater_than_50k']==False].hours_per_week.values,10,facecolor='green', alpha=0.5)
plt.xlabel('US citizens with income < $50K')
plt.ylabel('Hours worked per week')
#dist_data = pd.concat([data[data['greater_than_50k'] == True].groupby('hours_per_week')['hours_per_week'].count(),data[data['greater_than_50k'] == False].groupby('hours_per_week')['hours_per_week'].count()],axis=1)
#dist_data.columns=['gt50','lt50']
#dist_data_final = 100.0*dist_data['gt50']/(dist_data['gt50']+dist_data['lt50'])
#dist_data_final.sort(ascending=False)

#ax=dist_data_final.plot(kind='bar',color='r',y='Percentage')
#ax.set_xticklabels(dist_data_final.index,rotation=30,fontsize=8,ha='right')
#ax.set_xlabel('Hours worked per week')
#ax.set_ylabel('US citizens with income > $50k [%]')

# Income depends on nationality
fig = plt.figure(10)
dist_data = pd.concat([data[data['greater_than_50k'] == True].groupby('native_country')['native_country'].count(),data[data['greater_than_50k'] == False].groupby('native_country')['native_country'].count()],axis=1)
dist_data.columns=['gt50','lt50']
dist_data_final = 100.0*dist_data['gt50']/(dist_data['gt50']+dist_data['lt50'])
dist_data_final.sort(ascending=False)

ax=dist_data_final.plot(kind='bar',color='r')
ax.set_xticklabels(dist_data_final.index,rotation=40,fontsize=8,ha='right')
ax.set_xlabel('Native country')
ax.set_ylabel('US citizens with income > $50k [%]')

plt.show()
