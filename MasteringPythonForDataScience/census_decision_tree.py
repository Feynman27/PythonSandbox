#!/usr/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from patsy import dmatrices

data=pd.read_csv('../data/census.csv')
data=data.dropna(how='any')
data_test=pd.read_csv('../data/census_test.csv')
data_test=data_test.dropna(how='any')
# task: likelihood person will earn > $50K
# linear combination of variables
#formula='greater_than_50k ~ age+workclass+education+marital_status+occupation+race+gender+hours_per_week+native_country'
formula='greater_than_50k ~ education+workclass+age+hours_per_week+gender+occupation+race+marital_status+native_country'

# return outcome (y) and predictor (x) matrices
y_train,x_train = dmatrices(formula,data=data,return_type='dataframe')
y_test,x_test = dmatrices(formula,data=data_test,return_type='dataframe')

from sklearn import tree
clf=tree.DecisionTreeClassifier()
clf=clf.fit(x_train,y_train)

# Evaluate the DT performance on the test data
from sklearn.metrics import roc_curve,auc,classification_report
y_pred=clf.predict(x_test)
y_truth = y_test['greater_than_50k']
print pd.crosstab(y_truth, y_pred, rownames=['Truth'], colnames=['Predicted'])
print '\n\n'
print classification_report(y_truth,y_pred)

# ROC
fpr,tpr,thresholds=roc_curve(y_truth,y_pred)
roc_auc=auc(fpr,tpr)
print 'Area under curve: %f' % roc_auc
