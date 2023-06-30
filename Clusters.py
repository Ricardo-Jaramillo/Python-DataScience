from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from numpy.random import default_rng
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from Statistics import Stats
import seaborn as sns
import pandas as pd
import numpy as np
sns.set()


class Cluster:
    def __init__(self) -> None:
        pass


    def elbow_chart(self, dataset: pd.DataFrame, x_column: str, y_column: str, cluster_column: str=None):
        '''
        Plot the Elbow method chart to analyze the behavior of number of closters and the wcss parameter
        to look for the best/optimum number of clusters to be used in a KMeans clustering data.

        dataset:
            DataFrame containing the data to be plot
        x_column:
            Column name of the independent variabe
        y_column:
            Column name of the dependent variable
        cluster_columns:
            Column name of a categorical column to use for clustering the data
        '''

        dataset_copy = dataset.copy()

        #  Set the data to be Clustered from
        if cluster_column:
            data = pd.DataFrame(dataset_copy[cluster_column].astype('category').cat.codes)
            cluster_column = f'cluster_{cluster_column}'
        else:
            cluster_column = 'cluster'
            data = dataset_copy[[x_column, y_column]]
        
        # Create an empty list
        wcss=[]

        # Create all possible cluster solutions with a loop. max_clusters = n_rows
        max_clusters = data.shape[0] + 1
        
        for i in range(1, max_clusters):
            kmeans = KMeans(i)
            kmeans.fit(data)

            # Append the WCSS for the current iteration
            wcss_iter = kmeans.inertia_
            wcss.append(wcss_iter)
        
        # Plot
        clusters_number = range(1, max_clusters)
        
        plt.plot(clusters_number, wcss)
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('Within-cluster Sum of Squares')
        plt.show()


    def dendrogram(self, dataset: pd.DataFrame, x_column: str, y_column: str, plot: bool=False, n_clusters: int=0):
        '''
        Plot a dendrogram to look for an optimum number of clusters

        dataset:
            DataFrame containing the data to be clustered
        x_column:
            Column name of the independent variabe
        y_column:
            Column name of the dependent variable
        plot:
            Boolean that indicates plotting the results
        n_clusters:
            Number of desired clusters after checking the dendrogram plot
        '''
        
        # Set data to use for dendrogram
        data = dataset[[x_column, y_column]]
        
        # Plot if specified
        if plot:
            # Create dendrogram
            dendrogram = sch.dendrogram(sch.linkage(data, method = 'ward'))

            # Plot Scipy Dendrogram
            plt.title('Dendrogram')
            plt.xlabel(x_column)
            plt.ylabel('Euclidean distances')
            plt.show()

            # Plot seaborn heatmap
            # sns.clustermap(data, cmap='mako')
            # plt.show()
        
        # Get cluster array
        if n_clusters:
            hc = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
            data['Cluster'] = hc.fit_predict(data)
        
        return data
        

    def kmeans(self, dataset: pd.DataFrame, x_column: str, y_column: str, n_clusters: int, cluster_column=None, plot: bool=False, standarize_columns: list=[]):
        '''
        Clustering data with the KMeans method

        dataset:
            DataFrame containing the data to be clustered
        x_column:
            Column name of the independent variabe
        y_column:
            Column name of the dependent variable
        clusters:
            Number of desired clusters
        cluster_column:
            Categorical cluster Column name if specified
        plot:
            Boolean that indicates plotting the results
        standarize_columns:
            List containing column names to which data Mean and std will be standarize to 0 and 1, respectively.
            (Applies only to generate clustering, not to scale in the chart)
        
        Pros and Cons
        1. We need to specify K number of clusters - Elbow method
        2. Initialization sensitive - Use the solver
        3. Sensitive to outliers - Remove outliers
        4. Produces Spherical Solutions
        5. Standarization - Analyze the weights of the variables before/after standarization
        6. It can be usefull to prevent de Omitted viariable bias of the Regression Line
        '''
        
        dataset_copy = dataset.copy()
        
        # Set x, y for the Scatter Plot
        x = dataset_copy[x_column]
        y = dataset_copy[y_column]

        #  Set the data to be Clustered from
        if cluster_column:
            data = pd.DataFrame(dataset_copy[cluster_column].astype('category').cat.codes)
            cluster_column = f'cluster_{cluster_column}'
        else:
            cluster_column = 'cluster'
            data = dataset_copy[[x_column, y_column]]

        # Standarize data if specified
        if standarize_columns:
            stats = Stats()
            for column in standarize_columns:
                data[column] = stats.standarize_distribution(data[column])

        # Identify clusters
        kmeans = KMeans(n_clusters).fit(data)
        identified_clusters = kmeans.fit_predict(data)
        dataset_copy[cluster_column] = identified_clusters

        # Plot Clustered data
        if plot:
            plt.scatter(x, y, c=identified_clusters, cmap='rainbow')
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.title(f'{identified_clusters.max() + 1} Clusters')
            plt.show()

        return dataset_copy
    