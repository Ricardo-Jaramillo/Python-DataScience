from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.cluster import KMeans
from numpy.random import default_rng
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from matplotlib import style
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np
sns.set()


class Cluster:
    def __init__(self) -> None:
        pass


    def plot_elbow_chart(self, dataset: pd.DataFrame, x_column: str, y_column: str, cluster_column: str=None):
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


    def kmeans(self, dataset: pd.DataFrame, x_column: str, y_column: str, clusters: int, cluster_column=None, plot: bool=False):
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

        # Identify clusters
        kmeans = KMeans(clusters).fit(data)
        identified_clusters = kmeans.fit_predict(data)
        dataset_copy[cluster_column] = identified_clusters

        # Plot Clustered data
        if plot:
            plt.scatter(x, y, c=identified_clusters, cmap='rainbow')
            # plt.xlim(-180,180)
            # plt.ylim(-90,90)
            plt.show()

        return dataset_copy
    