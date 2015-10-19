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
import sklearn.ensemble as sk


# Evaluate the RF performance on the test data
from sklearn.metrics import roc_curve,auc,classification_report

# Improve model by requiring higher minimum (default=2) before splitting internal node
# and increasing minimum number of samples (default=1) in newly created leaves
clf=sk.RandomForestClassifier(n_estimators=100, oob_score=True, min_samples_split=5,min_samples_leaf=2)
clf=clf.fit(x_train,y_train['greater_than_50k'])
y_pred=clf.predict(x_test)
y_truth = y_test['greater_than_50k']
print pd.crosstab(y_truth, y_pred, rownames=['Truth'], colnames=['Predicted'])
print '\n\n'
print classification_report(y_truth,y_pred)

# ROC
fpr,tpr,thresholds=roc_curve(y_truth,y_pred)
roc_auc=auc(fpr,tpr)
print 'Area under curve: %f' % roc_auc

# Sort model features by importance
model_ranks = pd.Series(clf.feature_importances_, index=x_train.columns, name='Importance').sort(ascending=False,inplace=False)
model_ranks.index.name='Features'
# Top 30
top_features=model_ranks.iloc[:31].sort(ascending=True,inplace=False)
ifig = 1
fig = plt.figure(ifig, figsize=(15,7)); ifig+=1
ax = top_features.plot(kind='barh')
_ = ax.set_title("Variable Ranking")
_ = ax.set_xlabel('Performance Metric')
_ = ax.set_yticklabels(top_features.index,fontsize=8.0)
plt.show()
