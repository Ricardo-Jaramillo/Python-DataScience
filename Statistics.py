from numpy.random import default_rng
import matplotlib.pyplot as plt
from matplotlib import style
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np


# Init matplotlib style
style.use('ggplot')

# Setting a seed for random numbers
# print(SeedSequence().entropy)
rng = default_rng(122708692400277160069775657973126599887)


class Stats():
    def __init__(self):
        # self.data = data
        pass

    
    # n x 1 dataset
    def describe(self, data, sample=True, standarize=False):
        '''
        Empirical Rule for std -> 68-95-99.7

        # populate distribution with sample params
        data = stats.t.rvs(df=df, size=1000, scale=scale, loc=loc) #, random_state=rng)

        # Manually calculated
        skew = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        var = np.var(data, ddof=ddof)
        var_coeff = scale / loc
        '''

        if sample:
            ddof = 1
        else:
            ddof = 0

        if standarize:
            data = self.standarize_distribution(data)
        
        # Set distribution variables
        m, v, s, k = stats.t.stats(df=len(data) - 1, scale=np.std(data, ddof=ddof), loc=np.mean(data), moments='mvsk')
        std = v / (len(data) - ddof)
        vc = std / m

        # Set sample variables
        sn, (smin, smax), sm, sv, ss, sk = stats.describe(data, ddof=ddof)
        sstd = np.sqrt(sv)
        svc = sstd / sm

        # Save in a DataFrame
        dic = {
            'distribution': {
                'n': sn,
                'mean': m,
                'var': v,
                'skew': s,
                'kurtosis': k,
                'std': std,
                'var coefficient': vc
            },
            'sample': {
                'n': sn,
                'mean': sm,
                'var': sv,
                'skew': ss,
                'kurtosis': sk,
                'std': sstd,
                'var coefficient': svc
            }
        }

        df_desc = pd.DataFrame(dic)

        return df_desc
    

    # n x m DataFrame. Return an n x m cov_matrix and corr_coeff matrix.
    def cov_corr(self, data):
        
        cov_matrix = np.cov(data)
        corr_coeff_matrix = np.corrcoef(data)
        
        # Covariance Matrix
        sns.heatmap(cov_matrix, annot=True, fmt='g', xticklabels=data.index.to_list(), yticklabels=data.index.to_list())
        plt.title('Covariance Matrix')
        plt.tight_layout()
        plt.show()

        # Correlation Coefficient Matrix
        sns.heatmap(corr_coeff_matrix, annot=True, fmt='g', xticklabels=data.index.to_list(), yticklabels=data.index.to_list())
        plt.title('Correlation coefficient Matrix')
        plt.tight_layout()
        plt.show()

        return (cov_matrix, corr_coeff_matrix)
    

    # n x 2 dataset. y_actual, y_predicted order
    def confussion_matrix(self, y_actual, y_predicted, plot=True):
        '''
            Working with non numeric data
            df['y_actual'] = df['y_actual'].map({'Yes': 1, 'No': 0})
            df['y_predicted'] = df['y_predicted'].map({'Yes': 1, 'No': 0})
        '''
        confusion_matrix = np.array(pd.crosstab(y_actual, y_predicted, margins=False))
        
        if plot:
            sns.heatmap(confusion_matrix, annot=True)
            plt.show()

        return confusion_matrix
    

    # n x 1 dataset
    def standarize_distribution(self, data):
        return stats.zscore(data)


    # n x 1 dataset
    def central_limit(self, data, n_samples, frac):
        new_data = []

        # Make sure data is a DataFrame or pd Series
        if not (isinstance(data, pd.DataFrame) or isinstance(data, pd.Series)):
            data = pd.DataFrame(data=data, columns=['values'])
        
        # Generate n-sample means
        for i in range(n_samples):
            m = round(data.sample(frac=frac).mean(), 3)
            new_data.append(m)
        
        return new_data


    # n x 1 dataset or list of datasets with a desired confidence level. Must specify var population if known and assumed equal when required
    def confidence_interval(self, data, confidence, bilateral=True, var=None, var_assumed_equal=True, p_Test=None):
        
        # set confidence according to wheather it is a unilateral or bilateral test
        significance = 1 - confidence
        
        if bilateral:
            significance = significance / 2
            confidence = confidence + significance

        # If data contains a unique dataset. (Confidence Interval of a DataSet or Dependent samples)
        if not any(isinstance(el, list) for el in data):

            if var:
                critical_value = stats.norm.ppf(confidence)
            else:
                df = len(data)-1
                critical_value = stats.t.isf(significance, df)
                var = np.var(data, ddof=1)
                # With stats function
                # interval = stats.t.interval(confidence=confidence, df=len(data)-1, loc=np.mean(data), scale=stats.sem(data))
            
            m = np.mean(data)
            standard_error = np.sqrt(var / len(data))
        
        # If data is a list of datasets. (Independent samples)
        else:
            a = self.describe(data[0])['sample'].loc[['n', 'mean', 'var']]
            b = self.describe(data[1])['sample'].loc[['n', 'mean', 'var']]
            
            # Var known
            if var:
                m = a['mean'] - b['mean']
                critical_value = stats.norm.ppf(confidence)
                standard_error = np.sqrt(var[0] / a['n'] + var[1] / b['n'])

            # Var unknown but assumed equal
            elif var_assumed_equal:
                m = a['mean'] - b['mean']
                var = ((a['n'] - 1) * a['var'] + (b['n'] - 1) * b['var']) / (a['n'] + b['n'] - 2)
                df = a['n'] + b['n'] - 2
                
                critical_value = stats.t.isf(significance, df)
                standard_error = np.sqrt(var / a['n'] + var / b['n'])
            
            # Var unknown but assumed different
            else:
                m = a['mean'] - b['mean']
                df = ((a['var'] / a['n'] + b['var'] / b['n']) ** 2) / ((a['var'] / a['n']) ** 2 / (a['n'] - 1) + (b['var'] / b['n']) ** 2 / (b['n'] - 1))
                
                critical_value = stats.t.isf(significance, df)
                standard_error = np.sqrt(a['var'] / a['n'] + b['var'] / b['n'])

        margin_error = critical_value * standard_error
        low_lim = m - margin_error
        max_lim = m + margin_error
        
        print((low_lim, max_lim))
        # print(m, var, standard_error, critical_value)

        # If p_Test is required
        if p_Test != None:
            Z = abs((m - p_Test) / standard_error)

            # set p if var known and bilateral o unilateral Test
            if var:
                p = 1 - stats.norm.cdf(Z)
            else:
                p = 1 - stats.t.sf(Z)

            if bilateral:
                p *= 2

            # Evaluate Hypothesis Test
            op = lambda p, significance: '<' if p < significance else '>'
            print(f'{p} {op(p, significance)} {significance}: {p > significance} Hypothesis null')

        return (low_lim, max_lim)
    
    
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
        fig, ax = plt.subplots()
        sns.histplot(data=data_histogram, kde=kde, bins=bins, alpha=0.5)
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

    
    # n x 1 dataset
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
    