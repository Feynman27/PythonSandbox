#!/usr/bin/python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

'''
Logistic regression:
             1
    F(x) = -----
            1+exp^-(B_0+B_1*x)

Find B parameters that best fit:
    y=1 if B_0+B_1*x+epsilon>0
    y=0 otherwise
'''

df=pd.read_csv('../data/titanic_data.csv')

# count non-null values per feature
print df.count(0)

# remove Ticket,Cabin,and Name columns
df=df.drop(['Ticket','Cabin','Name'],axis=1)
# remove rows with null values
df=df.dropna()

# patsy: describes stat models (similar to R and S)
from patsy import dmatrices
# var left of '~' is dependent
# var right of '~' are indep
# C() treated as categorical vars
formula='Survived ~ C(Pclass) + C(Sex) + Age + SibSp + C(Embarked) + Parch'

# results dict that holds regression results for later analysis
# 600 rows, all columns
df_train=df.iloc[0:600,:]
# rows >=600, all columns
df_test=df.iloc[600:,:]

# Split into dep and indep vars
# return "y=outcome" and "x=predictor" matrices
y_train,x_train=dmatrices(formula,data=df_train,return_type='dataframe')
y_test,x_test=dmatrices(formula,data=df_test,return_type='dataframe')

# build model with statsmodels
import statsmodels.api as sm
model=sm.Logit(y_train,x_train)
res=model.fit()
# large p-values imply you cannot assume non-zero beta coefficients
# and hence, these vars are statistically insignificant
print res.summary()

# rebuild model with statistically significant variables i.e. p-values<0.05
formula = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp '
y_train,x_train = dmatrices(formula, data=df_train, return_type='dataframe')
y_test,x_test = dmatrices(formula, data=df_test, return_type='dataframe')

model = sm.Logit(y_train,x_train)
res = model.fit()
print res.summary()

# Model evaluation
# find pdf of predicted values
kde_res=sm.nonparametric.KDEUnivariate(res.predict())
kde_res.fit()
fig1,ax=plt.subplots(1,1)
plt.plot(kde_res.support,kde_res.density)
plt.fill_between(kde_res.support,kde_res.density, alpha=0.2)
plt.title('Distribution of survivor predictions')

# gender survivor segmentation
fig2,ax=plt.subplots(1,1)
plt.scatter(res.predict(),x_train['C(Sex)[T.male]'],alpha=0.2)
plt.grid(b=True,which='major',axis='x')
plt.xlabel("Predicted survival chance")
plt.ylabel("Male gender")
plt.title("Survival probability by gender (1=male, 0=female)")

# social class segmentation
fig3=plt.subplots(1,1)
plt.scatter(res.predict(),x_train['C(Pclass)[T.3]'] , alpha=0.2)
plt.xlabel("Predicted survival chance")
plt.ylabel("Class Bool (1=Third Class)")
plt.grid(b=True, which='major', axis='x')
plt.title("Surival probability by social class (1=3rd class, 0 = 1st/2nd class)")

# age segmentation
fig4=plt.subplots(1,1)
plt.scatter(res.predict(),x_train['Age'],alpha=0.2)
plt.grid(b=True, which='major', axis='x')
plt.xlabel("Predicted survival chance")
plt.ylabel("Age [yrs]")
plt.title("Surival probability by age")

# segmentation by no. of siblings/spouses
fig5=plt.subplots(1,1)
plt.scatter(res.predict(),x_train['SibSp'],alpha=0.2)
plt.grid(b=True, which='major', axis='x')
plt.xlabel("Predicted survival chance")
plt.ylabel("No. of siblings/spouses")
plt.title("Surival probability by no. of siblings/spouses")

# model performance via precision/recall
y_pred=res.predict(x_test)
# probability threshold
y_pred_flag = y_pred>0.7
print pd.crosstab(y_test.Survived,
                  y_pred_flag,
                  rownames=['Actual'],
                  colnames=['Predicted'])

print '\n\n'
from sklearn.metrics import roc_curve,auc,classification_report
# precision and recall
'''
precision = % of predicted values that are actually correct
recall = % of actual that was predicted correctly by the model
'''
print classification_report(y_test,y_pred_flag)

# compute ROC curve: TPR vs. FPR
'''
    FPR = FP/(FP+TN)
    TPR = TP/(TP+FN)
'''
fpr,tpr,thresholds=roc_curve(y_test,y_pred)
# area under curve
'''
    ROC = 1.0 is perfect
    ROC = 0.5 implies random guess
    Use traditional grading percentage system as accuracy metric
'''
roc_auc = auc(fpr, tpr)
print 'Area under curve: %f' % roc_auc

fig_roc_model1=plt.subplots(1,1)
plt.plot(fpr,tpr,label='ROC curve (area=%0.2f'%roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC)')
plt.legend(loc="lower right")

# build model with SciKit
# instantiate logistic regression model and fit with X and y
from sklearn import linear_model
model = linear_model.LogisticRegression()
model = model.fit(x_train,y_train.Survived)

# print beta coefficients
print pd.DataFrame(zip(x_train.columns, np.transpose(model.coef_)),columns=['Dep Vars','Beta_i'])

# precision and recall
y_pred=model.predict_proba(x_test)
y_pred_flag=y_pred[:,1]>0.7

print pd.crosstab(y_test.Survived,
                  y_pred_flag,
                  rownames=['Actual'],
                  colnames=['Predicted'])

print '\n\n'
print classification_report(y_test,y_pred_flag)

# ROC
fpr,tpr,thresholds=roc_curve(y_test,y_pred[:,1])
roc_auc = auc(fpr, tpr)
print 'Area under curve: %f' % roc_auc

fig_roc_model2=plt.subplots(1,1)
plt.plot(fpr,tpr,label='ROC curve (area=%0.2f'%roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC)')
plt.legend(loc="lower right")

plt.show()
