import pandas as pd
from numpy import nan as NA
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.mpl_style', 'default') # Make the graphs a bit prettier
plt.rcParams['figure.figsize'] = (15, 5)

# read in data set
dataset = pd.read_csv('/home/thomas/PythonSandbox/data/enigma-gov.eu.cfsp.sanctions.consolidated.entity-2cbbce78f53cbcd0225bf322c1dec87e.csv',na_values=[''])

dataset = dataset.dropna()
print dataset['id']
dataset['id'].plot()
