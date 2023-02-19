from SQLServer import SQLServer
import matplotlib.pyplot as plt
import pandas as pd


# Init SQLServer connection and get data
AdvWorks = SQLServer('AdventureWorks2019')
query = 'Select * from Sales.SalesOrderDetail'
sales = AdvWorks.select(query)


# How to group data
sales_grouped = sales[['ProductID', 'OrderQty', 'LineTotal']].groupby(['ProductID']).sum()


# How to create Pareto's Chart
    # Get Frequency of data and Sort
sales_frequency = sales[['ProductID', 'OrderQty']].rename(columns={'OrderQty': 'freq'}).groupby(['ProductID']).count().iloc[:10]
sales_frequency = sales_frequency.sort_values('freq', ascending=False)

    # Adj Index
ProductID = list(sales_frequency.index)
sales_frequency.index = sales_frequency.index.astype(str)

    # Get relative_frequency and totals
relative_frequency = sales_frequency['freq'].cumsum()
total_count = sales_frequency['freq'].sum()
sales_frequency['cum_freq'] = relative_frequency / total_count * 100

    # Plot Pareto's Chart
fig, ax = plt.subplots()
ax.bar(sales_frequency.index, sales_frequency["freq"], color="C0")
ax2 = ax.twinx()
ax2.plot(sales_frequency.index, sales_frequency["cum_freq"], color="C1", marker="o", ms=5)
plt.show() 


# How to Plot a Histogram
plt.hist(sales['ProductID'], bins = 10, color = "blue", rwidth=0.9)
plt.title("Product Hist")
plt.xlabel("Product")
plt.ylabel("Frequency")
plt.show()
