from SQLServer import SQLServer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import style
import seaborn as sns


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
data_histogram = pd.to_numeric(case['Case_HandleTimeHours'], errors='coerce').fillna(0).astype(float)

# Init matplotlib style
style.use('ggplot')

# Plot Histogram 1
bins = 7
plt.hist(data_histogram, bins=bins, color = "blue", rwidth=0.9, alpha=0.5)
# sales['ProductID'].plot.hist(alpha=0.5, bins=bins, grid=True, legend=None)
plt.title("Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")

# Plot Histogram 2
fig, ax = plt.subplots()
sns.histplot(data=data_histogram, x=data_histogram, kde=True, bins=bins)
ax.set_title('Histogram and KDE of sepal length')
plt.show()