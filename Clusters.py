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


    def kmeans(self, dataset: pd.DataFrame, x_column: str, y_column: str, clusters: int, cluster_column=None, plot: bool=False):
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