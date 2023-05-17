from SQLServer import SQLServer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import style
import seaborn as sns


# Init matplotlib style
style.use('ggplot')


class Stats():
    def __init__(self):
        # self.data = data
        pass

    
    # n x 2 Table with variable and 'freq' (freq must be in last position)
    def pareto(self, data, plot=False, xlim=False):
        field = data.columns.to_list()[0]

        # Group data by Frequency
        count = data.groupby([field]).count()

        # Order data
        data_frequency = count.sort_values('freq', ascending=False)#.iloc[5:15]

        # Get relative_frequency and totals
        origin_relative_frequency = data_frequency['freq'].cumsum()
        total_count = data_frequency['freq'].sum()
        data_frequency['cum_freq'] = origin_relative_frequency / total_count * 100
        # print(data_frequency)

        if plot:
            # Set fig and ax
            fig, ax = plt.subplots()
            ax2 = ax.twinx()
            
            # Plot
            ax.bar(data_frequency.index, data_frequency['freq'], color="C0")
            ax2.plot(data_frequency.index, data_frequency['cum_freq'], color="C1", marker="o", ms=5)
            
            # Set y lim from 0% - 100% and x ticks lim if requested
            plt.ylim([0, 110])
            if xlim:
                plt.xlim([-0.5, xlim + 0.5])
            
            # Rotate xticks
            # plt.xticks(xdata,catnames,rotation=90)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right') # plt.xticks()[1] -> another way to get the ticks
            plt.tight_layout()

            # Show values on the graph
            for i, v in enumerate(data_frequency['cum_freq']):
                plt.text(i, v, f'{str(int(v))}%', fontsize=8, color='black', verticalalignment='top')

            # Title Chart
            plt.title("Pareto's Chart")
            plt.xlabel("Product")
            plt.ylabel("Frequency")
            plt.show()    
        
        return data_frequency
    

    def histogram(self, data_histogram, bins):
        # Plot Histogram 1
        bins = 7
        plt.hist(data_histogram, bins=bins, color = "blue", rwidth=0.9, alpha=0.5)
        plt.title("Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")

        # Plot Histogram 2
        fig, ax = plt.subplots()
        sns.histplot(data=data_histogram, x=data_histogram, kde=True, bins=bins)
        ax.set_title('Histogram and KDE of sepal length')
        plt.show()

