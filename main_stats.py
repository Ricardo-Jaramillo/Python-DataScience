from sklearn import preprocessing
from SQLServer import SQLServer
from Statistics import Stats
import pandas as pd


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
stats = Stats()

# Describe & compare mvsk between distribution and Sample
# vars = stats.describe(case_CSATProduct['Case_CSAT'], sample=True, standarize=False)
# print(vars)

# Covariance and Correlation coefficient Matrix
# stats.cov_corr(case_metrics)

# Plot Confussion matrix
# stats.confussion_matrix(data_confussion_matrix['y_actual'], data_confussion_matrix['y_predicted'], plot=True)

# Standarize distribution
# a = stats.standarize_distribution(case_HandleTimeHours)
# print(case_HandleTimeHours)
# print(a)

# x_scaled = preprocessing.scale(case_HandleTimeHours)
# print(case_HandleTimeHours)
# print(x_scaled)

# Apply Central Limit Theorem
# a = stats.central_limit(case_HandleTimeHours, n_samples=1000, frac=0.6)
# print(a)

# Confidence Interval. Specify var if known. Specify Hypothesis Test if needed
# a = case_HandleTimeHours
# confidence = 0.90
# stats.confidence_interval(a, confidence, bilateral=False, var=None, var_assumed_equal=True, p_Test=None)

# Plot Paretos chart
# stats.pareto(case_DispositionReason, plot=True, xlim=False)

# Plot Histogram
# stats.histogram(case_HandleTimeHours, bins=10, kde=True)

# Plot Bars charts
# stats.bars(case_metrics, type='simple', stacked=False, rotation=0, table=False)
# stats.bars(case_metrics, type='simple', stacked=False, rotation=0, table=True)
# stats.bars(case_metrics, type='simple', stacked=True, rotation=0, table=False)
# stats.bars(case_metrics, type='simple', stacked=True, rotation=0, table=True)
# stats.bars(case_metrics, type='horizontal', stacked=False, rotation=0, table=False)
# stats.bars(case_metrics, type='horizontal', stacked=True, rotation=0, table=False)

# Plot probplot
# stats.probplot(case_HandleTimeHours)

# Plot scatter
# stats.scatter(case_scatter, colors=False, factor=10)
