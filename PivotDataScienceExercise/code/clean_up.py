import pandas as pd
from numpy import nan as NA
import numpy as np
import matplotlib.pyplot as plt

# Data from CE
consExpData = pd.read_csv('../Data_consumer_expenditure_survey.csv', na_values=['\N'])
print '=============================================================='
print 'Identifying columns with NULL values'
#print consExpData['emplcont'].isnull().sum()
# Count NULL values
# If NULL values > 30% of rows, delete column
max_null = 0.3*len(consExpData)
for column_name in consExpData.columns:
    if consExpData[column_name].isnull().sum() > max_null:
        print 'Deleting column named ' + str(column_name)
        del consExpData[column_name]
print '=============================================================='
df = consExpData.dropna()
print df.columns
print '=============================================================='
print '=============================================================='
print 'newid range: ' + str(df.newid.min()) + ' - ' + str(df.newid.max())
print '=============================================================='

db_test = df[df['newid']>=df.newid.median()]
## 70% of the data below the median
db_train = df[df['newid']<=df.newid.quantile(0.35)]
## validation set >35%ile but <50%ile
db_val2 = df[df['newid']>df.newid.quantile(0.35)]
db_val = db_val2[df['newid']<df.newid.median()]
print db_test['newid'].unique()
print db_train['newid'].unique()
print db_val['newid'].unique()

## Create, validate, and test model
import sklearn.ensemble as sk
rfc = sk.RandomForestClassifier(n_estimators=50, oob_score=True)
train_data = db_train[db_train.columns[1:]]
train_truth = db_train['wage']
model = rfc.fit(db_train,train_truth)
## out-of-bag score of training data set
print 'Out-of-bag score of training data set: ' + str(rfc.oob_score_)

