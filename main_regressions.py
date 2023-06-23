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
data_reg_3 = pd.read_csv('1.03. Dummies.csv')

# Set DataFrames

# Init Stats class
reg = Regressions()

# Transform variables
    # Apply log function
# data_reg_1_log = reg.transform(type='log', dataset=data_reg_1, columns=['SAT'])
# print(data_reg_1, data_reg_1_log)
    # Apply map function
# data_reg_3_map = reg.transform(type='map', dataset=data_reg_3, columns=['Attendance'])
# print(data_reg_3, data_reg_3_map)

# Plot a simple linear regression
# alpha = 0.05
# reg.linear_regression(dataset=data_reg_1, y_column='GPA', x_columns=['SAT'], alpha=alpha, dummy_columns=[])

# Multiple linear regression
# alpha = 0
# reg.linear_regression(dataset=data_reg_2, y_column='GPA', x_columns=['SAT', 'Rand'], dummy_columns=[], alpha=0)
# reg.linear_regression(dataset=data_reg_2, y_column='GPA', x_columns=['SAT'], dummy_columns=['Rand'], alpha=alpha)
# reg.linear_regression(dataset=data_reg_3, y_column='GPA', x_columns=['SAT', 'Attendance'], dummy_columns=[], alpha=0)
# reg.linear_regression(dataset=data_reg_3, y_column='GPA', x_columns=['SAT'], dummy_columns=['Attendance'], alpha=alpha)

# Make predictions with a Linear Regression
new_data_1 = pd.DataFrame({'SAT': [1700, 1670]})
new_data_2 = pd.DataFrame({'SAT': [1700, 1670], 'Attendance': ['No', 'Yes']})
new_data_3 = pd.DataFrame({'SAT': [1700, 1670], 'Attendance_No': [0, 1], 'Attendance_Yes': [1, 0]})
new_data = [new_data_2, new_data_3]

# To check
# ols_results = reg.linear_regression(dataset=data_reg_3, y_column='GPA', x_columns=['SAT'], dummy_columns=[], alpha=0)
# reg.predict(ols_results, [new_data_1])

# Correct
# ols_results = reg.linear_regression(dataset=data_reg_3, y_column='GPA', x_columns=['SAT', 'Attendance'], dummy_columns=[], alpha=0)
# reg.predict(ols_results, [new_data_2])

# To check
# ols_results = reg.linear_regression(dataset=data_reg_3, y_column='GPA', x_columns=['SAT'], dummy_columns=['Attendance'], alpha=0)
# reg.predict(ols_results, new_data)
