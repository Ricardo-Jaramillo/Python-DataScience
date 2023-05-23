from SQLServer import SQLServer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import style


# Init SQLServer connection and get data
Therabody = SQLServer('DbTherabody')
query = '''
    select
        Emp,
        agent_name,
        
        Date_Created,
        Date_Closed,
        Date_LastModified,
        Date_FirstResponseToCustomer,
        
        Case_Number,
        Case_RecordType,
        Case_Status,
        Case_Origin,
        Case_OriginAbs,
        Case_CSAT,
        Case_Disposition,
        Case_DispositionReason,
        Case_Disposition_Detailed,
        Case_Product,
        
        Case_FirstResponseToCustomerSeconds / 3600 Case_FRHours,
        Case_HandleTimeHours,
        Case_FRBusinessHours,
        1 as freq

    from V_Case
    where Date_Created >= '2023-01-01'
'''

# Set DataFrames
case = Therabody.select(query)
data_bars = case[[
                    'Case_CSAT',
                    'Case_FRHours', 
                    'Case_HandleTimeHours',
                    'Case_FRBusinessHours'
                ]].query('Case_CSAT != ""').replace('', np.nan)

# Init matplotlib style
style.use('ggplot')

# Group and sort data
data_bars = data_bars.groupby(['Case_CSAT']).mean() #.sort_values(by='Case_CSAT', ascending=False)

# Plot bars chart
    # Simple bars
array_xticks = np.arange(len(data_bars.index))
width = 0.2

plt.figure(figsize=(10, 5))
plt.bar(array_xticks, data_bars['Case_FRHours'], color='royalblue', width=width, label='FRT')
plt.bar(array_xticks + width, data_bars['Case_HandleTimeHours'], color='purple', width=width, label='AHT')
plt.bar(array_xticks - width, data_bars['Case_FRBusinessHours'], color='red', width=width, label='FRT BH')

plt.xticks(array_xticks, data_bars.index, rotation=90)
plt.tight_layout()
plt.title("Metrics by CSAT")
plt.xlabel("CSAT")
plt.ylabel("Min")
plt.legend()
plt.show()

    # Stacked bars
plt.figure(figsize=(10, 5))
plt.bar(array_xticks, data_bars['Case_FRHours'], color='royalblue', width=width, label='FRT')
plt.bar(array_xticks, data_bars['Case_HandleTimeHours'], color='purple', width=width, label='AHT', bottom=data_bars['Case_FRHours'])
plt.bar(array_xticks, data_bars['Case_FRBusinessHours'], color='red', width=width, label='FRT BH', bottom=data_bars['Case_HandleTimeHours'])

plt.xticks(array_xticks, data_bars.index, rotation=90)
plt.tight_layout()
plt.title("Metrics by CSAT")
plt.xlabel("CSAT")
plt.ylabel("Min")
plt.legend()
plt.show()

    # Stacked horizontal bars
plt.figure(figsize=(10, 5))
plt.barh(array_xticks, data_bars['Case_FRHours'], color='royalblue', height=width, label='FRT')
plt.barh(array_xticks, data_bars['Case_HandleTimeHours'], left=data_bars['Case_FRHours'], color='purple', height=width, label='Case_FRHours')

plt.yticks(array_xticks, data_bars.index)
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
