from SQLServer import SQLServer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import style
import seaborn as sns


# Init SQLServer connection and get data
AdvWorks = SQLServer('AdventureWorks2019')
query = 'Select * from V_sales_detailed'
sales = AdvWorks.select(query)


# Init matplotlib style
style.use('ggplot')


# Plot Histogram 1
bins = 7
plt.hist(sales['ProductID'], bins = bins, color = "blue", rwidth=0.9, alpha=0.5)
# sales['ProductID'].plot.hist(alpha=0.5, bins=bins, grid=True, legend=None)
plt.title("Product Histogram")
plt.xlabel("Product")
plt.ylabel("Frequency")


# Plot Histogram 2
fig, ax = plt.subplots()
sns.histplot(data=sales['ProductID'], x=sales['ProductID'], kde=True, bins=bins)
ax.set_title('Histogram and KDE of sepal length')
plt.show()