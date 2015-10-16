import pandas as pd
from numpy import nan as NA
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

#pd.set_option('display.mpl_style', 'default') # Make the graphs a bit prettier
#plt.rcParams['figure.figsize'] = (15, 5)

dataset = pd.read_csv('/home/thomas/PythonSandbox/data/Advertising.csv')
print dataset[:10]

# training set
X = dataset['TV']
y = dataset['Sales']
n_samples = int(len(X))

# %90 for training, 10% for testing
X_train = X[:int(0.9*n_samples)]
y_train = y[:int(0.9*n_samples)]
X_test = X[int(0.9*n_samples):]
y_test = y[int(0.9*n_samples):]
# must reshape to [n_samples,n_features]
# and [n_samples, n_targets] for X and y, respectively
X_train=X_train.reshape(len(X_train),1)
y_train=y_train.reshape(len(y_train),1)
X_test=X_test.reshape(len(X_test),1)
y_test=y_test.reshape(len(y_test),1)

# create linear regression object
regr = linear_model.LinearRegression()

# train model
regr.fit(X_train,y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)

# The mean square error
print("Residual sum of squares: %.2f"% np.mean((regr.predict(X_test) - y_test)**2 ) )

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, y_test))

# Plot outputs
fig = plt.figure(1, figsize=(4, 3))
plt.clf()
plt.scatter(X_test, y_test,  color='black')
plt.scatter(X_train, y_train,  color='blue')
plt.plot(X_test, regr.predict(X_test), color='blue', linewidth=3)
#plt.scatter(dataset['TV'],dataset['Sales'])
plt.xlim(-5.0,300.0)
plt.xlabel('TV Budget[thousands of $]')
plt.ylim(0.0,30.0)
plt.ylabel('Sales[thousands of $]')

plt.show()
