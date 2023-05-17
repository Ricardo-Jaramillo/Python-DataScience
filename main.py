from data_science import Stats
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
case_DispositionReason = case[['Case_DispositionReason', 'freq']] # .query('Case_DispositionReason != ""')
case_HandleTimeHours = pd.to_numeric(case['Case_HandleTimeHours'], errors='coerce').fillna(0).astype(float)

# Init Stats class
thera = Stats()

# Plot Paretos chart
thera.pareto(case_DispositionReason, plot=True)

# Plot Histogram
thera.histogram(case_HandleTimeHours, bins=10)
