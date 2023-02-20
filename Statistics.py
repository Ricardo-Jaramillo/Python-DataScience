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


# How to group data
sales_grouped = sales[['ProductName', 'OrderQty', 'LineTotal']].groupby(['ProductName']).sum()


# How to create Pareto's Chart
    # Get Frequency of data and Sort
sales_frequency = sales[['ProductName', 'OrderQty']].rename(columns={'OrderQty': 'freq'}).groupby(['ProductName']).count()
sales_frequency = sales_frequency.sort_values('freq', ascending=False).iloc[:5]

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


# How to Plot a Histogram
plt.figure(figsize=(10, 5))
plt.hist(sales['ProductID'], bins = 5, color = "blue", rwidth=0.9)
plt.title("Product Hist")
plt.xlabel("Product")
plt.ylabel("Frequency")
plt.show()


# How to plot a bar chart
    # Simple bars
sales_bars = sales_grouped.sort_values('OrderQty').iloc[:5]
array_xticks = np.arange(len(sales_bars.index))
width = 0.2

plt.figure(figsize=(10, 5))
plt.bar(array_xticks, sales_bars['OrderQty'], color='royalblue', width=width, label='Qty')
plt.bar(array_xticks + width, sales_bars['LineTotal'] / 100, color='purple', width=width, label='Total cost')

plt.xticks(array_xticks, sales_bars.index)
plt.title("Product Qty and Total cost")
plt.xlabel("Product")
plt.ylabel("Total")
plt.legend()
plt.show()

    # Stacked bars
sales_bars = sales_grouped.sort_values('OrderQty').iloc[:5]
array_xticks = np.arange(len(sales_bars.index))
width = 0.2

plt.figure(figsize=(10, 5))
plt.bar(array_xticks, sales_bars['OrderQty'], color='royalblue', width=width, label='Qty')
plt.bar(array_xticks, sales_bars['LineTotal'] / 100, bottom=sales_bars['OrderQty'], color='purple', width=width, label='Total cost')

plt.xticks(array_xticks, sales_bars.index)
plt.title("Product Qty and Total cost")
plt.xlabel("Product")
plt.ylabel("Total")
plt.legend()
plt.show()

    # Stacked horizontal bars
sales_bars = sales_grouped.sort_values('OrderQty').iloc[:5]
array_xticks = np.arange(len(sales_bars.index))
width = 0.2

plt.figure(figsize=(10, 5))
plt.barh(array_xticks, sales_bars['OrderQty'], color='royalblue', height=width, label='Qty')
plt.barh(array_xticks, sales_bars['LineTotal'] / 100, left=sales_bars['OrderQty'], color='purple', height=width, label='Total cost')

plt.yticks(array_xticks, sales_bars.index)
plt.title("Product Qty and Total cost")
plt.xlabel("Product")
plt.ylabel("Total")
plt.legend()
plt.show()