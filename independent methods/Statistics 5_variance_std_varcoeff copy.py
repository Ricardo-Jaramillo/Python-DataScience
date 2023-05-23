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
print(sales.describe())

# Init matplotlib style
style.use('ggplot')


# Print Mean and Std. Deviation. Describe Data
sales_grouped = sales.groupby(['ProductName']).sum()
print("Raw: %0.3f +/- %0.3f" % (np.mean(sales_grouped['LineTotal']), np.std(sales_grouped['LineTotal'])))


var = np.var(sales['LineTotal'])
std = np.std(sales['LineTotal'])
var_coeff = stats.variation(sales['LineTotal'])

print(var, std, var_coeff)