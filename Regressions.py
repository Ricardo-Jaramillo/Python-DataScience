from statsmodels.sandbox.regression.predstd import wls_prediction_std
from numpy.random import default_rng
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib import style
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np
sns.set()


# Init matplotlib style
style.use('ggplot')

# Setting a seed for random numbers
# print(SeedSequence().entropy)
rng = default_rng(122708692400277160069775657973126599887)


class Regressions():
    def __init__(self) -> None:
        pass

    
    # n x 2 DataFrame
    def simple_linear_regression(self, data: pd.DataFrame, alpha: float):
        
        # Get variable labels
        x_label = data.columns[0]
        y_label = data.columns[1]

        x1 = data[x_label]
        y = data[y_label]

        # Create the model
        x = sm.add_constant(x1)
        results = sm.OLS(y, x).fit()
        prstd, iv_l, iv_u = wls_prediction_std(results, alpha=alpha)
        
        # Get results
        summary = results.summary()
        params = results.params
        rsquared = results.rsquared
        
        print(summary)

        # Plot regression
        fig, ax = plt.subplots()

        ax.scatter(x1, y, color='blue')
        yhat = params[0] + params[1]*x1 # yhat = results.fittedvalues
        
        ax.plot(x1, yhat, c='orange', label ='regression line')
        ax.plot(x1, iv_u, 'r--')
        ax.plot(x1, iv_l, 'r--')
        
        plt.legend(loc='best')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
    