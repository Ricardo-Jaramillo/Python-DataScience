from Regressions import Regressions
from SQLServer import SQLServer
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

data_reg_1 = pd.read_csv('1.01. Simple linear regression.csv')
data_reg_2 = pd.read_csv('1.02. Multiple linear regression.csv')

# Set DataFrames
data_reg_1 = data_reg_1[[data_reg_1.columns[1], data_reg_1.columns[0]]]
data_reg_2 = data_reg_2[[data_reg_2.columns[1], data_reg_2.columns[0], data_reg_2.columns[2]]]

# Init Stats class
reg = Regressions()

# Transform variables with the log function
# data_reg_1_log = reg.transform_log(data_reg_1, ['SAT'])
# print(data_reg_1, data_reg_1_log)

# Plot a simple linear regression
# alpha = 0.05
# reg.linear_regression(data_reg_1, alpha)

# Multiple linear regression
# reg.linear_regression(data_reg_2)
