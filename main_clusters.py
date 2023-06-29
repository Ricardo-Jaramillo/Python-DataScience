from SQLServer import SQLServer
from Clusters import Cluster
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

data_1 = pd.read_csv('3.01. Country clusters.csv')

# Set DataFrames

# Init Regressions class
cluster = Cluster()

# Cluster data
clustered_data = cluster.kmeans(dataset=data_1, x_column='Longitude', y_column='Latitude', clusters=3, cluster_column=None, plot=True)
print(clustered_data)
# Cluster data with a categorical column
clustered_data = cluster.kmeans(dataset=data_1, x_column='Longitude', y_column='Latitude', clusters=3, cluster_column='Language', plot=True)
print(clustered_data)