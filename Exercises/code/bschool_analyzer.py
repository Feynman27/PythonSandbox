#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('../../data/US_BSchool_Rankings_2015.csv', na_values=['','-','N/A',' N/A ','  NaN'])
df.rename(columns=lambda x:x.strip())
col_names=[]
#for col in df.columns:
#    col_names.append(df[col][9])
col_names=['School','US News 15','Economist 14','BW 14','Forbes 13','FT 14','Class Size','Avg GMAT','Avg Age','Avg GPA','Acceptance rate','Full-time graduates employed within 3 Months','Average Salary', 'Average Signing Bonus']
df.columns=col_names
#print df[:20]
df=df[11:-1]
df=df.reset_index()
#df=df.fillna(-1)
print df[:20]

fig=plt.figure(1)
#df['School'] = df['School'].astype(str)
# convert to floats
for c in df.columns[2:]:
    df[c] = df[c].astype(float)
x=df['US News 15']
y=100.0*df['Acceptance rate']
plt.scatter(x,y,s=100)
plt.xlabel('Business school US News ranking')
plt.ylabel('Acceptance rate [%]')
plt.xlim([0.0,20.5])
plt.ylim([0.0,40.0])
for i,school in enumerate(df['School']):
    #print (school)
    #print str(x[i]) + str(y[i])
    plt.annotate(school,(x[i],y[i]),fontsize=11.5)
    if(i>20):break

fig=plt.figure(2)
y=df['Average Salary']+df['Average Signing Bonus']
plt.scatter(x,y,s=100)
plt.xlabel('Business school US News ranking')
plt.ylabel('Mean Salary (Base+Bonus) [USD]')
plt.xlim([0.0,20.5])
plt.ylim([100000.0,170000.0])
for i,school in enumerate(df['School']):
    if (school=='Duke University (Fuqua)'):
        plt.annotate(school,(x[i]+0.5,y[i]),fontsize=10)
    elif (school=='Cornell University (Johnson)'):
        plt.annotate(school,(x[i],y[i]+0.01*y[i]),fontsize=10)
    elif (school=='Carnegie Mellon University (Tepper)'):
        plt.annotate(school,(x[i]-0.5,y[i]+0.01*y[i]),fontsize=10)
    elif (i>0):
        if(abs(x[i]-x[i-1])/x[i]<0.01 ):
            plt.annotate(school,(x[i],y[i]-0.01*y[i]),fontsize=10)
        elif (abs(y[i]-y[i-1])/y[i]<0.01):
            plt.annotate(school,(x[i]-0.05*x[i],y[i]+0.01*y[i]),fontsize=10)
        else: plt.annotate(school,(x[i],y[i]),fontsize=10)
    else: plt.annotate(school,(x[i],y[i]),fontsize=10)
    if(i>20):break

plt.show()
