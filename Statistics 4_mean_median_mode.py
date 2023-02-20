from SQLServer import SQLServer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import style
from scipy import stats
import seaborn as sns


# Init SQLServer connection and get data
AdvWorks = SQLServer('AdventureWorks2019')
query = 'Select * from V_sales_detailed'
sales = AdvWorks.select(query)[['ProductID', 'TerritoryName', 'ProductName', 'OrderQty', 'LineTotal', 'SalesReasonName']]


# Init matplotlib style
style.use('ggplot')


# Print Mean and Std. Deviation. Describe Data
sales_grouped = sales.groupby(['ProductName']).sum()
print("Raw: %0.3f +/- %0.3f" % (np.mean(sales_grouped['LineTotal']), np.std(sales_grouped['LineTotal'])))


# Skew
print('\nSkewness for data : ', stats.skew(sales['ProductID']))


# Probability plot
fig, (ax) = plt.subplots(figsize = (4,4))
stats.probplot(sales['ProductID'],dist='norm', plot=ax)


# Plot Histogram 2
bins = 5
fig, ax = plt.subplots()
sns.histplot(data=sales['ProductID'], x=sales['ProductID'], kde=True, bins=bins)
ax.set_title('Histogram and KDE of sepal length')
plt.show()