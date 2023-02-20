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


# Plot bars chart
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
plt.ylabel("Product")
plt.xlabel("Total")
plt.legend()
plt.show()

    # How to plot bars chart with bottom table
data = [[ 66386, 174296,  75131, 577908,  32015],
        [ 58230, 381139,  78045,  99308, 160454],
        [ 89135,  80552, 152558, 497981, 603535],
        [ 78415,  81858, 150656, 193263,  69638],
        [139361, 331509, 343164, 781380,  52269]]

columns = ('Freeze', 'Wind', 'Flood', 'Quake', 'Hail')
rows = ['%d year' % x for x in (100, 50, 20, 10, 5)]

values = np.arange(0, 2500, 500)
value_increment = 1000

# Get some pastel shades for the colors
colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
n_rows = len(data)

index = np.arange(len(columns)) + 0.3
bar_width = 0.4

# Initialize the vertical-offset for the stacked bar chart.
y_offset = np.zeros(len(columns))

# Plot bars and create text labels for the table
cell_text = []
for row in range(n_rows):
    plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
    y_offset = y_offset + data[row]
    cell_text.append(['%1.1f' % (x / 1000.0) for x in y_offset])
# Reverse colors and text labels to display the last value at the top.
colors = colors[::-1]
cell_text.reverse()

# Add a table at the bottom of the axes
the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      rowColours=colors,
                      colLabels=columns,
                      loc='bottom')

# Adjust layout to make room for the table:
plt.subplots_adjust(left=0.2, bottom=0.2)

plt.ylabel(f"Loss in ${value_increment}'s")
plt.yticks(values * value_increment, ['%d' % val for val in values])
plt.xticks([])
plt.title('Loss by Disaster')
plt.show()