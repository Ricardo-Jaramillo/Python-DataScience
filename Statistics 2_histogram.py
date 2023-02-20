from SQLServer import SQLServer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import style


# Init SQLServer connection and get data
AdvWorks = SQLServer('AdventureWorks2019')
query = 'Select * from V_sales_detailed'
sales = AdvWorks.select(query)


# Init matplotlib style
style.use('ggplot')


# Plot Histogram
plt.figure(figsize=(10, 5))
plt.hist(sales['ProductID'], bins = 5, color = "blue", rwidth=0.9)
plt.title("Product Hist")
plt.xlabel("Product")
plt.ylabel("Frequency")
plt.show()
