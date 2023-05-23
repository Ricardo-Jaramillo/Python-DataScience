from SQLServer import SQLServer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import style
from scipy import stats
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
# case.to_csv('case.csv')
# case = pd.read_csv('case.csv')

data_histogram = case['Case_HandleTimeHours'][case['Case_HandleTimeHours'] <= 20] # [case['Case_CSAT' != ""]]

# Init matplotlib style
style.use('ggplot')

# Print Mean and Std. Deviation. Describe Data
print("Mean: %0.3f +/-std %0.3f" % (np.mean(data_histogram), np.std(data_histogram)))
print('Median: %0.3f' % np.median(data_histogram))

# Skew
print('\nSkewness for data : ', stats.skew(data_histogram))

# Probability plot
fig, (ax) = plt.subplots(figsize = (4,4))
stats.probplot(data_histogram,dist='norm', plot=ax)

# Plot Histogram 2
bins = 5
fig, ax = plt.subplots()
sns.histplot(data=data_histogram, x=data_histogram, kde=True, bins=bins)
ax.set_title('Histogram and KDE of sepal length')
plt.show()