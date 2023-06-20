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

    
    # n x m DataFrame
    def transform_log(self, data, columns):
        # Apply log function to the specified columns
        for col in columns:
            data[col] = np.log(data[col])
        
        return data

    
    # n x 2 DataFrame with first column y
    def linear_regression(self, data: pd.DataFrame, alpha=None):
        '''
        # Assumptions
        1. Linearity.
            Make sure transform the data if needed
        2. No endogeneity.
            Omitted variable bias
            Covariance between the error terms and the independent variables
        3. Normality and homoscedasticity.
            Homoscedasticity means equal variance along the regression line.
            Transform into the log variable
        4. No autocorrelation. Error terms are not correlated
        5. No multicollinearity. Independent variables have no correlation within each others.
            Omit those variables
        '''
        
        '''
        # Important variables
        P Value T Test. Test the significance of a variable
        R Squared. Explain the variability of the model
        Adj R Squared. Test the significance of more variables in a model
        F Value F Test. To compare the significance of the model within different models
        Durbin - Watson. 2 -> No autocorrelation, <1 and >3 sign to alarm
        '''

        # Get variable labels
        y_label = data.columns[0]
        x_label = data.columns[1:]

        x1 = data[x_label]
        y = data[y_label]

        # Create the model
        x = sm.add_constant(x1)
        results = sm.OLS(y, x).fit()
        
        # Get results
        summary = results.summary()
        params = results.params
        rsquared = results.rsquared
        
        print(summary)

        # Plot if simple linear regression
        if len(x_label) == 1:
            # Plot Regression
            fig, ax = plt.subplots()

            ax.scatter(x1, y, color='blue')
            yhat = params[0] + params[1]*x1 # yhat = results.fittedvalues
            
            ax.plot(x1, yhat, c='orange', label ='regression line')
            
            # Get confidence bounds
            if alpha:
                prstd, iv_l, iv_u = wls_prediction_std(results, alpha=alpha)
                ax.plot(x1, iv_u, 'r--')
                ax.plot(x1, iv_l, 'r--')
            
            plt.legend(loc='best')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.show()
        