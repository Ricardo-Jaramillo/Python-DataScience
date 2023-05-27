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
data = case[['Case_Product', 'Case_CSAT']]

# Init matplotlib style
style.use('ggplot')

data_grouped = data.groupby(['Case_Product']).mean()

var = np.var(data_grouped['Case_CSAT'])
std = np.std(data_grouped['Case_CSAT'])
var_coeff = stats.variation(data_grouped['Case_CSAT'])

print(var, std, var_coeff)