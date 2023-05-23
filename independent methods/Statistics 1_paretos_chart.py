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
        
        Case_FirstResponseToCustomerSeconds,
        Case_HandleTimeHours,
        Case_FRBusinessHours,
        1 as freq

    from V_Case
    where Date_Created >= '2023-01-01'
'''

# Set DataFrames
case = Therabody.select(query)
case_origin = case[['Case_OriginAbs', 'freq']]

# Init matplotlib style
style.use('ggplot')

# Group data by Frequency
origin_count = case_origin.groupby(['Case_OriginAbs']).count()
# print(origin_count)

# Order data
origin_frequency = origin_count.sort_values('freq', ascending=False)#.iloc[5:15]
print(origin_frequency)

# Get relative_frequency and totals
origin_relative_frequency = origin_frequency['freq'].cumsum()
total_count = origin_frequency['freq'].sum()
origin_frequency['cum_freq'] = origin_relative_frequency / total_count * 100
print(origin_frequency)

# Plot Pareto's Chart
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.bar(origin_frequency.index, origin_frequency["freq"], color="C0")
ax2.plot(origin_frequency.index, origin_frequency["cum_freq"], color="C1", marker="o", ms=5)

plt.ylim([0, 110])
plt.title("Pareto's Chart")
plt.xlabel("Product")
plt.ylabel("Frequency")
plt.show()
