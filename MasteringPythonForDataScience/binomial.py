#!/usr/bin/python

from scipy.stats import binom
import matplotlib.pyplot as plt

fig,ax = plt.subplots(3,sharex=True, sharey=True)
try:
    trials=int(raw_input('Enter number of trials:'))
except ValueError:
    print "Enter a number."

x = range(trials+1)

n,p = trials,0.5

#rv = binom(n,p)
prob=[0.4,0.5,0.6]
for i in range(0,len(prob)):
    rv=binom(n,prob[i])
    ax[i].vlines(x, 0, rv.pmf(x), colors='k', linestyles='-', lw=1,label='Probablity='+str(prob[i]))
    ax[i].legend(loc='best', frameon=False)

#ax.vlines(x, 0, rv.pmf(x), colors='k', linestyles='-', lw=1,label='Probablity')
#ax.legend(loc='best', frameon=False)

plt.show()
