from sklearn import preprocessing
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
data_2 = pd.read_csv ('3.12. Example.csv')
data_2_scaled = pd.DataFrame(preprocessing.scale(data_2), columns=[data_2.columns.to_list()])
data_3 = pd.read_csv('Country clusters standardized.csv', index_col='Country')

# Set DataFrames
data_3 = data_3.drop(['Language'],axis=1)
# Init Regressions class
cluster = Cluster()

# Plot Elbow Chart to find the optimum number of clusters
# cluster.elbow_chart(dataset=data_1, x_column='Longitude', y_column='Latitude', cluster_column=None)
# cluster.elbow_chart(dataset=data_1, x_column='Longitude', y_column='Latitude', cluster_column='Language')

# Create a dendrogram to find the optimum number of clusters
# cluster.dendrogram(dataset=data_3, x_column='Longitude', y_column='Latitude', plot=True, n_clusters=0)
# data_3 = cluster.dendrogram(dataset=data_3, x_column='Longitude', y_column='Latitude', plot=False, n_clusters=3)
# print(data_3)

# Cluster data with standarized transformation
# clustered_data = cluster.kmeans(dataset=data_2, x_column='Satisfaction', y_column='Loyalty', n_clusters=3, cluster_column=None, plot=True, standarize_columns=['Satisfaction'])

# Cluster data
# clustered_data = cluster.kmeans(dataset=data_1, x_column='Longitude', y_column='Latitude', n_clusters=3, cluster_column=None, plot=False)
# print(clustered_data)

# Cluster data with categorical column
# clustered_data = cluster.kmeans(dataset=data_1, x_column='Longitude', y_column='Latitude', n_clusters=3, cluster_column='Language', plot=False)
# print(clustered_data)
