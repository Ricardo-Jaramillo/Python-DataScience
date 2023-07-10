from Regressions import Regressions
from SQLServer import SQLServer
import pandas as pd
import os

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
path = os.path.join(os.path.abspath(os.getcwd()), 'datasets')

data_reg_1 = pd.read_csv(f'{path}/1.01. Simple linear regression.csv')
data_reg_2 = pd.read_csv(f'{path}/1.02. Multiple linear regression.csv')
data_reg_3 = pd.read_csv(f'{path}/1.03. Dummies.csv')
data_reg_4 = pd.read_csv(f'{path}/2.01. Admittance.csv')
data_reg_5 = pd.read_csv(f'{path}/2.02. Binary predictors.csv')

# Set DataFrames
data_reg_3['Attendance'] = data_reg_3['Attendance'].map({'Yes': 1, 'No': 0})
data_reg_4['Admitted'] = data_reg_4['Admitted'].map({'Yes': 1, 'No': 0})
data_reg_5['Admitted'] = data_reg_5['Admitted'].map({'Yes': 1, 'No': 0})
data_reg_5['Gender'] = data_reg_5['Gender'].map({'Female': 1, 'Male': 0})

# Init Regressions class
reg = Regressions()

# Plot a simple linear regression
# alpha = 0.05
# reg.regression_model(type='linear', dataset=data_reg_1, y_column='GPA', x_columns=['SAT'], alpha=alpha, dummy_column=None, plot=True)

# Multiple linear regression
# alpha = 0.0
# reg.regression_model(type='linear', dataset=data_reg_2, y_column='GPA', x_columns=['SAT', 'Rand'], alpha=alpha, dummy_column=None, plot=True)
# reg.regression_model(type='linear', dataset=data_reg_2, y_column='GPA', x_columns=['SAT', 'Rand'], alpha=alpha, dummy_column='Rand', plot=True)
# reg.regression_model(type='linear', dataset=data_reg_3, y_column='GPA', x_columns=['SAT', 'Attendance'], alpha=alpha, dummy_column=None, plot=True)
# reg.regression_model(type='linear', dataset=data_reg_3, y_column='GPA', x_columns=['SAT', 'Attendance'], alpha=alpha, dummy_column='Attendance', plot=True)

# Make predictions with a Linear Regression
# new_data_1 = pd.DataFrame({'SAT': [1700, 1670]})
# new_data_2 = pd.DataFrame({'SAT': [1700, 1670], 'Attendance': [0, 1]})

# To check
# ols_results = reg.regression_model(type='linear', dataset=data_reg_1, y_column='GPA', x_columns=['SAT'], alpha=0, dummy_column=None, plot=True)
# reg.predict(ols_results['Original'], new_data_1)

# Correct
# ols_results = reg.regression_model(type='linear', dataset=data_reg_3, y_column='GPA', x_columns=['SAT', 'Attendance'], alpha=0, dummy_column=None, plot=True)
# reg.predict(ols_results['Original'], new_data_2)

# To check
# ols_results = reg.regression_model(type='linear', dataset=data_reg_3, y_column='GPA', x_columns=['SAT', 'Attendance'], alpha=0, dummy_column='Attendance', plot=True)
# reg.predict(ols_results['Original'], new_data_2)
# reg.predict(ols_results['Dummy'], new_data_2)

# Logistic Regression
# reg.regression_model(type='logistic', dataset=data_reg_4, y_column='Admitted', x_columns=['SAT'], plot=False)
# reg.regression_model(type='logistic', dataset=data_reg_5, y_column='Admitted', x_columns=['SAT', 'Gender'], plot=False)