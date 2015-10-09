#/usr/bin/python
from scipy import stats
import matplotlib.pyplot as plt

#fig,ax = plt.subplots(1,1)
# Bernoulli trial with 70% success rate
print stats.bernoulli.rvs(0.7, size=100)
