from SQLServer import SQLServer
from DeepLearning import DeepLearning
import pandas as pd
import numpy as np

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
training_data = np.load('TF_intro.npz')

# Set DataFrames

# An exercise with the form y = 2x - 3z + 5
# observations = 1000
# xs = np.random.uniform(low=-10, high=10, size=(observations,1))
# zs = np.random.uniform(-10, 10, (observations,1))
# generated_inputs = np.column_stack((xs,zs))
# noise = np.random.uniform(-1, 1, (observations,1))
# generated_targets = 2*xs - 3*zs + 5 + noise
# np.savez('TF_intro', inputs=generated_inputs, targets=generated_targets)

# Init Regressions class
deep_learning = DeepLearning()
