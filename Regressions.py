from statsmodels.sandbox.regression.predstd import wls_prediction_std
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


# Init matplotlib style
style.use('ggplot')

# Setting a seed for random numbers
# print(SeedSequence().entropy)
rng = default_rng(122708692400277160069775657973126599887)


class Regressions():
    def __init__(self) -> None:
        self.color = [
                        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',  
                        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
                    ]

    def __transform(self, type: str, dataset: pd.DataFrame, columns: list):
        '''
        Private method to transform DataFrame columns
        type:
            log -> Applies a logarithmic function to data
            map -> Mapps a color to each group of integer values of the specified columns to plot and observe in a chart
        dataset:        
            n x m DataFrame
        columns:
            Specified columns of the dataset to transform
        '''

        data_copy = dataset.copy()
        # Apply a transformation for each column
        for column in columns:
            # Apply log function to the specified columns
            if type == 'log':
                data_copy[column] = np.log(dataset[column])
            # Map data 
            if type == 'map':
                i = 0
                # Iterate value replacement with each color
                for val in dataset[column].unique():
                    data_copy[column] = data_copy[column].replace(val, self.color[i])
                    i += 1
        
        return data_copy
            

    def __plot_linear_regression(self, dataset: pd.DataFrame, y_column: str, x_column: str, models: dict, alpha: float=0, dummy_column: str=None):
        '''
        Private method to plot a simple linear regression with maximum 1 Dummy variable
        dataset:
            n x m Dataframe with m total variables to include in the Linear model Regression
        y_column:
            Column name of the dependent variable
        x_column:
            Column name of the independent variable
        models:
            Dictionary that contains OLS fitted model results from statsmodels.formula.api.
            It can contains only the simple model or both the simple and the one that includes a dummy variable
        alpha:
            Significance level to which bounds of the model will be plot
        dummy_column:
            Column name of the Dummy variable

        Assumptions
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

        Important variables in the Summary Regression model
        P Value T Test -> Test the significance of a variable
        R Squared -> Explain the variability of the model
        Adj R Squared -> Test the significance of more variables in a model
        F Value F Test -> To compare the significance of the model within different models
        Durbin - Watson -> No autocorrelation, <1 and >3 sign of alarm
        '''

        # Init variables data
        y = dataset[y_column]
        x = dataset[x_column]

        # Set colors
        fig, ax = plt.subplots()
        ax.set_prop_cycle(color=self.color)

        if dummy_column:
            c = self.__transform(type='map', dataset=dataset, columns=[dummy_column])[dummy_column]
        else:
            c = self.color[1]
        
        # Plot Scatter x, y
        fig = ax.scatter(x, y, c=c)

        # For each model
        for model_type, model_results in models.items():
            # Get model params
            params = model_results.params
            
            # Original Regression Line Equation
            if model_type == 'Original':
                # Plot Regression Line
                yhat = params[0] + params[1]*x
                fig = ax.plot(x, yhat, label='Regression line')

                # Plot confidence bounds
                if alpha:
                    prstd, iv_l, iv_u = wls_prediction_std(model_results, alpha=alpha)
                    fig = ax.plot(x, iv_u, 'r--', label='upper bound')
                    fig = ax.plot(x, iv_l, 'r--', label='lower bound')

            # Dummmy model Regression Line Equations
            if model_type == 'Dummy' and dummy_column:
                for value in dataset[dummy_column].unique():
                    # Plot each dummy Regression Line
                    yhat_dummies = params['Intercept'] + params[dummy_column]*value + params[x_column]*x
                    fig = ax.plot(x, yhat_dummies, label=f'{dummy_column} {value}')

        plt.title('Simple linear Regression')
        plt.legend(loc='best')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.show()


    def __plot_logistic_regression(self, dataset: pd.DataFrame, y_column: str, x_column: str, model_results: smf):
        '''
        Private method to plot a Logistic Regression
        dataset:
            n rows x 2 columns Dataframe
        y_column:
            Column name of the dependent variable
        x_column:
            Column name of the independent variable
        model_results:
            Logit fitted model results from statsmodels.formula.api
        '''

        # Get model params and variables data
        b0, b1 = model_results.params

        y = dataset[y_column]
        x = dataset[x_column]
        x_sort = np.sort(np.array(x))
        yhat = np.sort(np.array(np.exp(b0 + x*b1) / (1 + np.exp(b0 + x*b1))))
        
        fig, ax = plt.subplots()
        ax.set_prop_cycle(color=self.color)
        
        ax.scatter(x, y, c=self.color[0])
        
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.plot(x_sort, yhat, c=self.color[1])
        plt.show()


    def regression_model(self, type: str, dataset: pd.DataFrame, y_column: str, x_columns: list, alpha: float=0, dummy_column: str=None, plot: bool=True):
        '''
        Main method to create the selected Regression model
        type:
            One of linear, logistic.
        dataset:
            n x m DataFrame that contains y and x columns to include in the model Regression
        y_column:
            Column name of the dependent variable
        x_columns:
            Column name of the independent variables that the model will process. It must contains the x and Dummy variable if dummy_column specified.
        alpha:
            Significance level to which bounds of the model will be plot
        dummy_column:
            Column name of the Dummy variable (Must be specified in the x_columns too)
        plot:
            boolean that allows to plot the model (It only works for a simple Linear Regression model, i.e. if the remaining x_columns is equals to 1 unique x variable: x_columns - dummy_column == 1)
        '''

        # Set the model expression
        reg_exp = f'{y_column} ~ {" + ".join(x_columns)}'
        
        # Create the selected model
        if type == 'linear':
            models = {}
            model_results = smf.ols(formula=reg_exp, data=dataset).fit()
            print(model_results.summary())
            
            # If dummy then append Dummy model and re-create the Original model
            if dummy_column:
                models['Dummy'] = model_results
                x_columns.remove(dummy_column)

                reg_exp = f'{y_column} ~ {" + ".join(x_columns)}'
                model_results = smf.ols(formula=reg_exp, data=dataset).fit()
                print(model_results.summary())

            # Append the original model to the models list
            models['Original'] = model_results
            
            # Plot if remains a unique x_column variable
            if plot and len(x_columns) == 1:
                self.__plot_linear_regression(dataset, y_column, x_columns[0], models, alpha, dummy_column)

        if type == 'logistic':
            model_results = smf.logit(formula=reg_exp, data=dataset).fit()
            print(model_results.summary())

            # Plot if alpha is specified
            if plot:
                self.__plot_logistic_regression(dataset, y_column, x_columns[0], model_results)

        return models


    def predict(self, results: smf, dataset: pd.DataFrame):
        '''
        Method to make predictions according to a Fitted Regression model
        results:
            Regression fitted model results from statsmodels.formula.api
        dataset:
            n x m DataFrame from which it may be required to make predictions.
            Dataset contains the same number of x_columns as the fitted model.
        '''

        # Predict and join to the same dataset
        predictions = results.predict(dataset)
        dataset = dataset.join(pd.DataFrame({'Predictions': predictions}))
        print(dataset)

        return predictions
