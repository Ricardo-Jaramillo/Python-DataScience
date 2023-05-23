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
        
        Case_FirstResponseToCustomerSeconds / 3600 Case_FRHours,
        Case_HandleTimeHours,
        Case_FRBusinessHours,
        1 as freq

    from V_Case
    where Date_Created >= '2023-01-01'
'''

# Reach data
# case = Therabody.select(query)
# case.to_csv('case.csv')
case = pd.read_csv('case.csv')

# Set DataFrames
case_DispositionReason = case[['Case_DispositionReason', 'freq']].groupby(['Case_DispositionReason']).count() # .query('Case_DispositionReason != ""')

case_HandleTimeHours = case['Case_HandleTimeHours'][case['Case_HandleTimeHours'] <= 20]

case_metrics = case[[
                    'Case_CSAT',
                    'Case_HandleTimeHours',
                    'Case_FRBusinessHours',
                    'Case_FRHours'
                ]].query('Case_CSAT != ""').replace('', np.nan).groupby(['Case_CSAT']).mean()

# Init Stats class
thera = Stats()

# Plot Paretos chart
thera.pareto(case_DispositionReason, plot=True, xlim=False)

# Plot Histogram
thera.histogram(case_HandleTimeHours, bins=10, kde=True)

# Plot Bars charts
thera.bars(case_metrics, type='simple', stacked=False, rotation=0, table=False)
thera.bars(case_metrics, type='simple', stacked=False, rotation=0, table=True)
thera.bars(case_metrics, type='simple', stacked=True, rotation=0, table=False)
thera.bars(case_metrics, type='simple', stacked=True, rotation=0, table=True)
thera.bars(case_metrics, type='horizontal', stacked=False, rotation=0, table=False)
thera.bars(case_metrics, type='horizontal', stacked=False, rotation=0, table=True)
thera.bars(case_metrics, type='horizontal', stacked=True, rotation=0, table=False)
thera.bars(case_metrics, type='horizontal', stacked=True, rotation=0, table=True)

# Describe mean, std, median and skew
thera.skew(case_HandleTimeHours)

# Plot probplot
thera.probplot(case_HandleTimeHours)