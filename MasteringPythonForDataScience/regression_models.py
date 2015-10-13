#!/usr/bin/python

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import linear_model
import matplotlib.pyplot as plt

sl_data=pd.read_csv('../data/Mens_height_weight.csv')
fig1,ax=plt.subplots(1,1)
ax.scatter(sl_data['Height'],sl_data['Weight'])
ax.set_xlabel('Ht.[cm]')
ax.set_ylabel('Wt.[kg]')

# correlation matrix
print sl_data.corr()


######################
# Linear  Regression #
######################

# Create linear regression object
lm=linear_model.LinearRegression()

# train model using training set
# weight = dep, height= indep variables
lm.fit(sl_data.Height[:,np.newaxis],sl_data.Weight)

print '\n'
print 'Intercept is: ' + str(lm.intercept_) + '\n'
print 'Coefficient value of the height is: ' + str(lm.coef_) + '\n'
# create df of indep var
print pd.DataFrame(zip(sl_data.columns,lm.coef_),columns=['features','estimatedCoefficients'])
print '\n'

fig2,ax=plt.subplots(1,1)
# x,y scatter plot
ax.scatter(sl_data.Height,sl_data.Weight)
# plot fit result
ax.plot(sl_data.Height, lm.predict(sl_data.Height[:, np.newaxis]),color = 'red')
ax.set_xlabel('Ht.[cm]')
ax.set_ylabel('Wt.[kg]')

######################
# Multiple Regression#
######################
# Y = a + Sum b_n*x_n

# Predict avg pts scored per NBA game
b_data = pd.read_csv('../data/basketball.csv')
print '#############'
print '#  NBA Data #'
print '#############'
# print statistics
print b_data.describe()
print '\n'
# print correlation matrix
print 'Printing correlation matrix:'
print b_data.corr()
print '\n'

fig3,ax=plt.subplots(1,1)
ax.scatter(b_data.height,b_data.avg_points_scored)
ax.set_xlabel('height')
ax.set_ylabel('Avg. points per game')

fig4,ax=plt.subplots(1,1)
ax.scatter(b_data.weight,b_data.avg_points_scored)
ax.set_xlabel('weight')
ax.set_ylabel('Avg. points per game')

fig5,ax=plt.subplots(1,1)
ax.scatter(b_data.success_field_goals,b_data.avg_points_scored)
ax.set_xlabel('FG percentage')
ax.set_ylabel('Avg. points per game')

fig6,ax=plt.subplots(1,1)
ax.scatter(b_data.success_free_throws,b_data.avg_points_scored)
ax.set_xlabel('FT percentage')
ax.set_ylabel('Avg. points per game')

############################
# Training and testing data #
############################

from sklearn import cross_validation,feature_selection,preprocessing
import statsmodels.formula.api as sm
from statsmodels.tools.eval_measures import mse
from statsmodels.tools.tools import add_constant
from sklearn.metrics import mean_squared_error

# convert to array structure
X=b_data.values.copy()

# split into random train/test sets (80:20)
# Dependent variable is avg_points_scored
# Indep vars = X[:,:-1]; means all rows and all columns except the last column
# Dep vars = X[:,-1] means all rows and only the last column
X_train,X_valid,y_train,y_valid = cross_validation.train_test_split( X[:,:-1], X[:,-1],train_size=0.80)

# Ordinary Least Squares (OLS) regression
# add_constant() calculates intercept (by default, not performed)
# p-value<0.05 implies significant variable
result = sm.OLS(y_train,add_constant(X_train) ).fit()
print '\n'
print 'Printing OLS fit summary: '
print result.summary()
print '\n'

# use different combination of variables
print '\n'
print 'Printing alternative OLS fit summary: '
result_alt = sm.OLS(y_train,add_constant(X_train[:,2]) ).fit()
print result_alt.summary()
print '\n'

# apply model on test data set
ypred = result.predict(add_constant(X_valid) )
print 'Printing mean squared error: '
# mean-square error
print mse(ypred,y_valid)
print '\n'

# predict test set
ypred_alt = result_alt.predict(add_constant(X_valid[:,2]))
print 'Printing mean squared error of alternate model: '
print mse(ypred_alt,y_valid)
print '\n'

fig7,ax=plt.subplots(1,1)
ax.scatter(y_valid,ypred)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')

# alternate model
fig8,ax=plt.subplots(1,1)
ax.scatter(y_valid,ypred_alt)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')

# Train model using training sets
lm = linear_model.LinearRegression()
lm.fit(X_train,y_train)
print 'Intercept is %f' % lm.intercept_
print pd.DataFrame(zip(b_data.columns,lm.coef_),columns=['features','estimatedCoefficients'])

# R^2: runs 3 times for cross-validation
print 'R^2:' + str(cross_validation.cross_val_score(lm,X_train,y_train,scoring='r2'))
# apply model on test data set
ypred=lm.predict(X_valid)
# MSE
print 'MSE: ' + str( mean_squared_error(ypred,y_valid)  )

fig9,ax=plt.subplots(1,1)
ax.scatter(y_valid,ypred)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')

# plot figures
plt.show()
