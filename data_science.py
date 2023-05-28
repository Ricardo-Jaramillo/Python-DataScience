import matplotlib.pyplot as plt
from SQLServer import SQLServer
from matplotlib import style
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np


# Init matplotlib style
style.use('ggplot')


class DataScience():
    def __init__(self):
        # self.data = data
        pass


    # n x 1 dataset
    def skew(self, data):
        # Print Mean and Std. Deviation. Describe Data
        mean = np.mean(data)
        std = np.std(data)
        median = np.median(data)

        print('Mean: %0.3f +/-std %0.3f' % (mean, std))
        print('Median: %0.3f' % median)

        # Skew
        skew = stats.skew(data)
        print('\nSkewness for data : ', skew)

        return (mean, median, std, skew)

    
        # n x 1 column dataset
    def var(self, data, sample=False):
        '''
        Empirical Rule
        68-95-99.7
        '''
        if sample:
            ddof = 1
        else:
            ddof = 0

        mean = np.mean(data)
        var = np.var(data, ddof=ddof)
        std = np.std(data, ddof=ddof)
        var_coeff = std / mean

        print(f'mean: {mean}')
        print(f'var: {var}')
        print(f'std: {std}')
        print(f'var_coeff: {var_coeff}')

        return (mean, var, std, var_coeff)
    
    
    # n x 2 Table with variable name and 'freq' grouped by variable name (freq must be in last position)
    def pareto(self, data, plot=False, xlim=False):
        # Order data
        data_frequency = data.sort_values('freq', ascending=False)#.iloc[5:15]

        # Get relative_frequency and totals
        origin_relative_frequency = data_frequency['freq'].cumsum()
        total_count = data_frequency['freq'].sum()
        data_frequency['cum_freq'] = origin_relative_frequency / total_count * 100
        print(data_frequency)

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
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8) # plt.xticks()[1] -> another way to get the ticks
            # plt.tight_layout()
            plt.subplots_adjust(left=0.2, bottom=0.4)

            # Show values on the graph
            for i, v in enumerate(data_frequency['cum_freq']):
                plt.text(i, v, f'{str(int(v))}%', fontsize=8, color='black', verticalalignment='top')

            # Title Chart
            plt.title("Pareto's Chart")
            plt.xlabel("Product")
            plt.ylabel("Frequency")
            plt.show()    
        
        return data_frequency
    
    
    # list with values to plot
    def histogram(self, data_histogram, bins, kde=False):
        # Histogram type 1
        # plt.hist(data_histogram, bins=bins, color = "blue", rwidth=0.9, alpha=0.5)
        # plt.title("Histogram")
        # plt.xlabel("Value")
        # plt.ylabel("Frequency")

        # Histogram type 2
        fig, ax = plt.subplots()
        sns.histplot(data=data_histogram, x=data_histogram, kde=kde, bins=bins, alpha=0.5)
        ax.set_title('Histogram')
        plt.show()


    # dataframe (max 7 columns)
    def bars(self, data_bars, type='simple', stacked=False, rotation=0, table=False):
        # init variables
        array_xticks = np.arange(len(data_bars.index))
        prev_data_bars = np.zeros(len(data_bars.index))
        y_offset = prev_data_bars
        width = 0.1
        cell_text = []

        if stacked:
            aux = 0
        else:
            aux = width
        
        offset = np.array([0, 1, -1, 2, -2, 3, -3]) * aux
        # color = ['blue', 'purple', 'red', 'green', 'yellow', 'cian', 'orange']
        
        # plot according scpecs
        for i in range(len(data_bars.columns)):
            column = data_bars.columns[i]

            if type == 'simple':
                plt.bar(array_xticks + offset[i], data_bars[column], width=width, label=column, bottom=prev_data_bars)
                plt.setp([plt.xticks()[1]], rotation=rotation, ha='right') # , fontsize=8)
                plt.ylabel("Value")

                if not table:
                    plt.xlabel(f"{data_bars.index.name}")
            
            elif type == 'horizontal':
                plt.barh(array_xticks + offset[i], data_bars[column], height=width, label=column, left=prev_data_bars)
                plt.xlabel("Value")
                plt.ylabel(f"{data_bars.index.name}")
                plt.yticks(array_xticks, data_bars.index)
            
            if stacked:
                prev_data_bars += np.array(data_bars[column])

            if table and type != 'horizontal':
                if stacked:
                    y_offset = prev_data_bars
                else:
                    y_offset = data_bars[column]
                cell_text.append(['%1.1f' % x for x in y_offset])
                plt.xticks([])

        if table and type != 'horizontal':
            # cell_text.reverse()
            plt.table(cellText=cell_text, rowLabels=data_bars.columns.to_list(), colLabels=array_xticks.tolist(), loc='bottom') # rowColours=colors, 
            plt.subplots_adjust(left=0.2, bottom=0.2)
        
        plt.title(f"{data_bars.index.name} {type} bars")
        plt.tight_layout() # plt.subplots_adjust(bottom=0.2)
        plt.legend()
        plt.show()

    
    def probplot(self, data):
        '''
        ppplot (Probability-Probability plot)
            Compares the sample and theoretical probabilities (percentiles).
        
        qqplot (Quantile-Quantile plot)
            Compares the sample and theoretical quantiles
        
        probplot (Probability plot)
            Same as a Q-Q plot, however probabilities are shown in the scale of the theoretical distribution (x-axis) and the y-axis contains unscaled quantiles of the sample data.
        '''
        fig, (ax) = plt.subplots(figsize = (4,4))
        stats.probplot(data, dist='norm', plot=ax)
        plt.show()

    
    # n x 2 DataFrame (n x 3 if area passed to method) following the order: x, y, area (area optional)
    def scatter(self, data, colors=False, factor=1):
        try:
            area = data[data.columns[2]] * factor
        except:
            area = None

        if colors:
            colors = np.arange(len(data.index))
        else:
            colors = None

        plt.scatter(data[data.columns[0]], data[data.columns[1]], s=area, c=colors, alpha=0.5)
        plt.show()
    

