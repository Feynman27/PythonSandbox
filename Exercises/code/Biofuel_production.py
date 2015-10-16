import pandas as pd
from numpy import nan as NA
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.mpl_style', 'default') # Make the graphs a bit prettier
plt.rcParams['figure.figsize'] = (15, 5)

# read in data set
raw_dataset = pd.read_csv('/home/thomas/PythonSandbox/data/enigma-com.bp.statistical-review.biofuels-production-ktoe-d390d4e6c70f05d89ddfdbcf75dc456f.csv',na_values=[''])

# Plot the transpose
# Keep all rows and remove columns 1 and the last two
raw_dataset_transpose = raw_dataset.iloc[:,1:-3].T
raw_dataset_transpose.plot()
#dataset = dataset.dropna()
#print dataset
#print dataset['']
#dataset['id'].plot()
