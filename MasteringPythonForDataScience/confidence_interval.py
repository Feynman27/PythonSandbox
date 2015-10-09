#!/usr/bin/python

''' Average height of Kenyans '''

average_height=[]
try:
    trials = int(raw_input('Enter the number of trials: '))
    sample_size = int(raw_input('Enter the sample size: '))
except ValueError:
    print 'Enter a number.'

import numpy as np
# Generage normal distribution of heights for each trial
for i in xrange(trials):
    sample50 = np.random.normal(183.0,10,sample_size).round()
    average_height.append(sample50.mean())

import matplotlib.pyplot as plt
plt.hist(average_height,10,normed=True)
plt.show()
