from SQLServer import SQLServer
import matplotlib.pyplot as plt


# Init SQLServer connection and get data
AdvWorks = SQLServer('AdventureWorks2019')

query = 'Select * from Sales.SalesOrderDetail'

sales = AdvWorks.select(query)

# How to group data
sales_grouped = sales[['ProductID', 'OrderQty', 'LineTotal']].groupby(['ProductID']).sum()

# 1_ Plot a Histogram
plt.hist(sales['ProductID'], bins = 10, color = "blue", rwidth=0.9)
plt.title("Product Hist")
plt.xlabel("Product")
plt.ylabel("Frequency")
plt.show()

