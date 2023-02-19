from SQLServer import SQLServer


AdvWorks = SQLServer('AdventureWorks2019')

query = 'Select * from Sales.SalesOrderDetail'

response = AdvWorks.select(query)
print(response)