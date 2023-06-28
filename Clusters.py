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


    def kmeans(self, dataset, x_column, y_column, clusters):
        # Set x, y for the Scatter Plot and the data to be Clustered
        x = dataset[x_column]
        y = dataset[y_column]
        data = dataset[[x_column, y_column]]

        # Identify clusters
        kmeans = KMeans(clusters).fit(data)
        identified_clusters = kmeans.fit_predict(data)

        # Plot Clustered data
        plt.scatter(x, y, c=identified_clusters, cmap='rainbow')
        
        plt.xlim(-180,180)
        plt.ylim(-90,90)
        plt.show()