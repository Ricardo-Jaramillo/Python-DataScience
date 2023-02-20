from SQLServer import SQLServer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import style


# Init SQLServer connection and get data
AdvWorks = SQLServer('AdventureWorks2019')
query = 'Select * from V_sales_detailed'
sales = AdvWorks.select(query)[['ProductID', 'TerritoryName', 'ProductName', 'OrderQty', 'LineTotal', 'SalesReasonName']]
print(sales.describe())


# Init matplotlib style
style.use('ggplot')


# Print Mean and Std. Deviation. Describe Data
sales_grouped = sales.groupby(['ProductName']).sum()
print("Raw: %0.3f +/- %0.3f" % (np.mean(sales_grouped['LineTotal']), np.std(sales_grouped['LineTotal'])))
print(sales_grouped.describe())


# Plot Histogram
df = sales['ProductID']
df.plot.hist(alpha=0.5, bins=10, grid=True, legend=None)
plt.xlabel("ProductID")
plt.title("Histogram")
plt.show()


# Apply a function
df_exp = df.apply(np.log)   # pd.DataFrame.apply accepts a function to apply to each column of the data
df_exp.plot.hist(alpha=0.5, bins=10, grid=True, legend=None)
plt.xlabel("Feature value")
plt.title("Histogram")
plt.show()


from sklearn import preprocessing

X_s = preprocessing.StandardScaler().fit_transform(df_exp)
X_s = pd.DataFrame(X_s)   # Put the np array back into a pandas DataFrame for later
print("StandardScaler: %0.3f +/- %0.3f" % (np.mean(X_s), np.std(X_s)))

# Nice! This should be 0.000 +/- 1.000
# StandardScaler: -0.000 +/- 1.000