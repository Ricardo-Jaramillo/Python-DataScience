from data_science import DataScience
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
    where Date_Created >= '2023-06-01'
'''

# Reach data
case = Therabody.select(query)
# case.to_csv('case.csv')
# case = pd.read_csv('case.csv')

# Set DataFrames
case_DispositionReason = case[['Case_DispositionReason', 'freq']].groupby(['Case_DispositionReason']).count() # .query('Case_DispositionReason != ""')

case_HandleTimeHours = case['Case_HandleTimeHours'][(case['Case_HandleTimeHours'] <= 20) & (case['Case_HandleTimeHours'] > 0)]

case_metrics = case[[
                    'Case_Product',
                    'Case_CSAT',
                    'Case_HandleTimeHours',
                    'Case_FRBusinessHours',
                    'Case_FRHours'
                ]].query('Case_CSAT != ""').groupby(['Case_Product']).mean()

case_scatter = case[['Date_Created', 'Case_FRBusinessHours', 'Case_CSAT']]

case_CSATProduct = case[['Case_Product', 'Case_CSAT']].groupby(['Case_Product']).mean()

# Confussion Matrix
data = {'y_actual':    [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        'y_predicted': [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0]
        }
data_confussion_matrix = pd.DataFrame(data)

# Init Stats class
thera = DataScience()

# Describe & compare mvsk between distribution and Sample
# thera.describe(case_CSATProduct['Case_CSAT'], sample=True, standarize=False)

# Covariance and Correlation coefficient Matrix
# thera.cov_corr(case_metrics)

# Plot Confussion matrix
# thera.confussion_matrix(data_confussion_matrix, plot=True)

# Standarize distribution
# case_HandleTimeHours = thera.standarize_distribution(case_HandleTimeHours)

# Apply Central Limit Theorem
# case_HandleTimeHours = thera.central_limit(case_HandleTimeHours, n_samples=1000, frac=0.6)

# Confidence Interval. Specify var if known. Specify Hypothesis Test if needed
# a = case_HandleTimeHours
# confidence = 0.90
# thera.confidence_interval(a, confidence, bilateral=False, var=None, var_assumed_equal=True, p_Test=None)

# Plot Paretos chart
# thera.pareto(case_DispositionReason, plot=True, xlim=False)

# Plot Histogram
# thera.histogram(case_HandleTimeHours, bins=10, kde=True)

# Plot Bars charts
# thera.bars(case_metrics, type='simple', stacked=False, rotation=0, table=False)
# thera.bars(case_metrics, type='simple', stacked=False, rotation=0, table=True)
# thera.bars(case_metrics, type='simple', stacked=True, rotation=0, table=False)
# thera.bars(case_metrics, type='simple', stacked=True, rotation=0, table=True)
# thera.bars(case_metrics, type='horizontal', stacked=False, rotation=0, table=False)
# thera.bars(case_metrics, type='horizontal', stacked=True, rotation=0, table=False)

# Plot probplot
# thera.probplot(case_HandleTimeHours)

# Plot scatter
# thera.scatter(case_scatter, colors=False, factor=10)

