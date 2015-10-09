from scipy.stats import poisson
import matplotlib.pyplot as plt
import numpy as np

fig,ax = plt.subplots(3,sharex=True, sharey=True)

my_lambda = [1,4,10]

for i in range(0,len(my_lambda)):
    mu=my_lambda[i]
    x = np.arange(poisson.ppf(0.01, mu),poisson.ppf(0.99, mu))
    rv=poisson(mu)
    #ax[i].vlines(x, 0, poisson.pmf(x, mu), colors='b', lw=5, alpha=0.5)
    ax[i].vlines(x, 0, rv.pmf(x), colors='b', lw=5,alpha=0.5, label='Lambda='+str(mu))
    ax[i].legend(loc='best', frameon=False)

plt.show()

try:
    mean=int(raw_input('Enter the expectation value lambda for a Poisson distribution: '))
except ValueError:
    print 'Enter a number.'

rv=poisson(mean)

try:
    k=int(raw_input('Enter the number of occurences you wish to find the probability for: '))
except ValueError:
    print 'Enter a number.'
print 'The probability of obtaining ' + str(k) + ' occurences with E(X) = ' + str(mean) + ' is P(X=k) = ' + str(rv.pmf(k))
