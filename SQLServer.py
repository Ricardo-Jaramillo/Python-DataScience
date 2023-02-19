import requests
import pyodbc

cnxn_str = ("Driver={SQL Server Native Client 11.0};"
            "Server=localhost"
            "Database=AdventureWorks2019;"
            "Trusted_Connection=yes;")

cnxn = pyodbc.connect(cnxn_str)

query = 'Select * from Select * from Sales.SalesOrderDetail'

response = requests.get(query, cnxn)

print(response)