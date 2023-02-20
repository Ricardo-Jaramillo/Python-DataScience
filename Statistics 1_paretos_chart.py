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


# Group data
sales_grouped = sales[['ProductName', 'OrderQty', 'LineTotal']].groupby(['ProductName']).sum()


# Create Pareto's Chart
    # Get Frequency of data and Sort
sales_frequency = sales[['ProductName', 'OrderQty']].rename(columns={'OrderQty': 'freq'}).groupby(['ProductName']).count()
sales_frequency = sales_frequency.sort_values('freq', ascending=False).iloc[5:15]

    # Adj Index
sales_frequency.index = sales_frequency.index.astype(str)

    # Get relative_frequency and totals
relative_frequency = sales_frequency['freq'].cumsum()
total_count = sales_frequency['freq'].sum()
sales_frequency['cum_freq'] = relative_frequency / total_count * 100

    # Plot Pareto's Chart
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.bar(sales_frequency.index, sales_frequency["freq"], color="C0")
ax2.plot(sales_frequency.index, sales_frequency["cum_freq"], color="C1", marker="o", ms=5)

plt.title("Pareto's Chart")
plt.xlabel("Product")
plt.ylabel("Frequency")
plt.show() 
