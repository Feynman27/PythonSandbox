#!/usr/bin/python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv('/home/thomas/PythonSandbox/data/titanic_train.csv')

# make column headers all lowercase
#df.columns = map(str.lower,df.columns)

######################################
## Survivors by social class (1,2,3)##
######################################
print df['Pclass'].isnull().value_counts()

# Passengers that survived per social class
survivors = df.groupby('Pclass')['Survived'].agg(sum)

# Total passengers per social class
total_passengers = df.groupby('Pclass')['PassengerId'].count()
survivor_percentage = survivors/total_passengers

# plotting
fig1=plt.figure(1)
ax=fig1.add_subplot(111)
rect=ax.bar(survivors.index.values.tolist(), survivors, color='blue', width=0.5)
ax.set_ylabel('# of survivors')
ax.set_title('Total # of survivors per social class')
xTickMarks=survivors.index.values.tolist()
ax.set_xticks(survivors.index.values.tolist())
xtickNames=ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames,fontsize=20)
#plt.show(block=True)


# Plot survival rate per class
fig2=plt.figure(2)
ax2 = fig2.add_subplot(111)

rect2 = ax2.bar(survivor_percentage.index.values.tolist(), survivor_percentage*100.0, color='blue', width=0.5)
ax2.set_ylabel('Survivor Rate[%]')
ax2.set_title('Survival Rate by Social Class')
xTickMarks2=survivors.index.values.tolist()
ax2.set_xticks(survivors.index.values.tolist())
xtickNames2=ax2.set_xticklabels(xTickMarks2)
plt.setp(xtickNames2,fontsize=20)
#plt.show(block=True)

########################
## Survivors by gender##
########################

# Check for null values
print df['Sex'].isnull().value_counts()

# Male-passenger survivors per class
male_survivors = df[df['Sex']=='male'].groupby('Pclass')['Survived'].agg(sum)

# Total male passengers per class
male_total_passengers = df[df['Sex']=='male'].groupby('Pclass')['PassengerId'].count()
male_survivor_percentage = male_survivors/male_total_passengers

# Female-passenger survivors per class
female_survivors = df[df['Sex']=='female'].groupby('Pclass')['Survived'].agg(sum)

# Total female passengers per class
female_total_passengers = df[df['Sex']=='female'].groupby('Pclass')['PassengerId'].count()
female_survivor_percentage = female_survivors/female_total_passengers

# Plotting by gender
fig = plt.figure(3)
ax = fig.add_subplot(111)
index = np.arange(male_survivors.count())
bar_width = 0.35
rect1 = ax.bar(index, male_survivors, bar_width, color='blue', label='Men')
rect2 = ax.bar(index + bar_width, female_survivors, bar_width, color='y', label='Women')
ax.set_ylabel('Survivor Numbers')
ax.set_title('# of Survivors by Social Class')
xTickMarks = male_survivors.index.values.tolist()
ax.set_xticks(index + bar_width)
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, fontsize=20)
plt.legend()
plt.tight_layout()
#plt.show(block=True)

# Plot survival rate per class by gender
fig2=plt.figure(4)
ax2 = fig2.add_subplot(111)
index = np.arange(male_survivors.count())
bar_width = 0.35
rect1 = ax2.bar(index, male_survivor_percentage*100.0,bar_width, color='blue',  label='male')
rect2 = ax2.bar(index+bar_width, female_survivor_percentage*100.0,bar_width, color='y', label='female')
ax2.set_ylabel('Survivor Rate[%]')
ax2.set_title('Survival Rate by Social Class')
xTickMarks2 = male_survivors.index.values.tolist()
ax2.set_xticks(index + bar_width)
xtickNames2 = ax2.set_xticklabels(xTickMarks2)
plt.setp(xtickNames, fontsize=20)
plt.legend()
plt.tight_layout()
#plt.show()
plt.close()

###############################################
## Non-survivors with family members onboard ##
###############################################

# Check for null values for:
# siblings/spouses aboard
print df['SibSp'].isnull().value_counts()
# parents/children aboard
print df['Parch'].isnull().value_counts()

# total no. of non-survivors per social class
non_survivors=df[ ( (df['SibSp']>0) | (df['Parch']>0) ) & (df['Survived']==0) ].groupby('Pclass')['Survived'].agg('count')

# total passengers per class
total_passengers = df.groupby('Pclass')['PassengerId'].count()

non_survivor_percentage = non_survivors/total_passengers

# plot non-survivors with family onboard by social class
fig5 = plt.figure(5)
ax = fig5.add_subplot(111)
rect = ax.bar(non_survivors.index.values.tolist(), non_survivors, color='blue', width=0.5)
ax.set_ylabel('No. of non-survivors with family onboard')
ax.set_title('Total number of non-survivors with family onboard by social class')
xTickMarks = non_survivors.index.values.tolist()
ax.set_xticks(non_survivors.index.values.tolist())
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, fontsize=20)

# Percentage of non-survivors with family onboard by social class
fig6 = plt.figure(6)
ax = fig6.add_subplot(111)
rect = ax.bar(non_survivor_percentage.index.values.tolist(), non_survivor_percentage*100.0, color='blue', width=0.5)
ax.set_ylabel('Non-Survivors with family onboard [%]')
ax.set_title('Non-survivors with family onboard by social class')
xTickMarks = non_survivor_percentage.index.values.tolist()
ax.set_xticks(non_survivor_percentage.index.values.tolist())
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, fontsize=20)

##########################
## Survival rate by age ##
##########################

# Check null values
print df['Age'].isnull().value_counts()

# age intervals
age_bins=[0,18,25,40,60,110]

df['AgeBin']=pd.cut(df.Age,bins=age_bins)

# remove null values
d_temp = df[np.isfinite(df['Age'])] # removes all na instances

# number of survivors grouped by age
survivors=d_temp.groupby('AgeBin')['Survived'].agg(sum)

# total passengers per age group
total_passengers=d_temp.groupby('AgeBin')['PassengerId'].agg('count')

# survival rates
survivor_age_percentage = survivors/total_passengers

# plot a pie graph
fig7 = plt.figure(7)
ax = fig7.add_subplot(111)
plt.pie(total_passengers,labels=total_passengers.index.values.tolist(),autopct='%1.1f%%', shadow=True,startangle=90)
plt.title('Total passengers by age')

# plot pie graph of survivors by age group
fig8 = plt.figure(8)
ax = fig8.add_subplot(111)
plt.pie(survivors, labels=survivors.index.values.tolist(),autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('Survivals (relative to all passengers) by age')

# plot histogram of survival rate by age group
fig9 = plt.figure(9)
ax = fig9.add_subplot(111)
rect = ax.bar(range(len(survivor_age_percentage)), survivor_age_percentage*100.0, color='blue', width=0.5)
ax.set_ylabel('Survival rate [%]')
ax.set_title('Survival rate by age group')
xTickMarks = range(len(survivor_age_percentage))
ax.set_xticks(range(len(survivor_age_percentage)))
xtickNames = ax.set_xticklabels(survivor_age_percentage.index.values.tolist())
plt.setp(xtickNames, fontsize=20)
plt.show()
