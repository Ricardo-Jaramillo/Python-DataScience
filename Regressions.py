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

    # n x m DataFrame. Specify the columns to transform
    def transform(self, type: str, dataset: pd.DataFrame, columns: list):
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
    

    # Plot a simple Regression with dummy variables if specified
    def __plot_simple_linear_regression(self, dataset, model_results, y_column, x_column, x_column_dummies, dummy_columns, alpha):
        # Set scatter colors if dummies
        if dummy_columns:
            c = dataset[dummy_columns[0]]
        else:
            c = self.color[1]
        
        y = dataset[y_column]
        x = dataset[x_column]
        
        # Plot x, y
        fig, ax = plt.subplots()
        ax.set_prop_cycle(color=self.color)
        
        fig = ax.scatter(x, y, c=c)
        
        # Plot Regression looping for model (&) dummy model
        for model_type, model in model_results:
            # Get params and Init counter_color
            params = model.params
            rsquared = model.rsquared

            # Original Regression Line Equation
            if model_type == 'Original':
                # Set Regression Line
                yhat = params[0] + params[x_column]*dataset[x_column]
                
                # Plot regression line
                fig = ax.plot(x, yhat, label='Regression line')

                # Plot confidence bounds
                if alpha:
                    prstd, iv_l, iv_u = wls_prediction_std(model, alpha=alpha)
                    fig = ax.plot(x, iv_u, 'r--', label='upper bound')
                    fig = ax.plot(x, iv_l, 'r--', label='lower bound')

            # Dummy_model Regression Line Equations
            if model_type == 'Dummy':
                # Set Regression Line for each dummy variable
                for column in dataset[x_column_dummies]:
                    yhat_dummies = params[0] + params[column] + params[x_column]*dataset[x_column]
                    
                    # Plot each dummy Regression Line
                    fig = ax.plot(x, yhat_dummies, label=column)
            
        plt.title('Simple linear Regression')
        plt.legend(loc='best')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.show()
        

    def __plot_linear_regression(self, dataset: pd.DataFrame, y_column: str, x_column: str, models: dict, alpha: float=0, dummy_column: str=None):
        # Init variables data
        y = dataset[y_column]
        x = dataset[x_column]

        # Set colors
        fig, ax = plt.subplots()
        ax.set_prop_cycle(color=self.color)

        if dummy_column:
            values = dataset[dummy_column].unique()
            dataset['colors'] = dataset[dummy_column]

            for value in values:
                dataset['colors'] = dataset['colors'].replace(value, self.color[value])
            c = dataset['colors']
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
                # Set Regression Line
                yhat = params[0] + params[1]*x
            
                # Plot Regression Line
                fig = ax.plot(x, yhat, label='Regression line')

                # Plot confidence bounds
                if alpha:
                    prstd, iv_l, iv_u = wls_prediction_std(model_results, alpha=alpha)
                    fig = ax.plot(x, iv_u, 'r--', label='upper bound')
                    fig = ax.plot(x, iv_l, 'r--', label='lower bound')

            # Dummmy model Regression Line Equations
            if model_type == 'Dummy' and dummy_column:
                for value in dataset[dummy_column].unique():
                    yhat_dummies = params['Intercept'] + params[dummy_column]*value + params[x_column]*x
                        
                    # Plot each dummy Regression Line
                    fig = ax.plot(x, yhat_dummies, label=f'{dummy_column} {value}')

        plt.title('Simple linear Regression')
        plt.legend(loc='best')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.show()


    def __plot_logistic_regression(self, dataset: pd.DataFrame, y_column: str, x_column: str, model_results: smf):
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


    # n x 2 DataFrame (y, xn-1, xn)
    def linear_regression(self, dataset: pd.DataFrame, y_column: str, x_columns: list, alpha: float=0, dummy_columns: list=[]):
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
        
        # Init Results list and dataset with dummies
        aux = dataset[dummy_columns]
        
        results = []
        dataset = pd.get_dummies(data=dataset, columns=dummy_columns, dtype=float)

        x_columns_dummies = []
        for dummy in dummy_columns:
            # x_columns_plot = dataset.loc[:, [dummy not in item and y_column != item for item in dataset.columns]].columns.to_list()
            x_columns_dummies = dataset.loc[:, [dummy in item for item in dataset.columns]].columns.to_list()
        
        # Map dummy columns
        dataset[dummy_columns] = self.transform('map', aux, dummy_columns)

        # Formulate the original model
        reg_exp = f'{y_column} ~ {" + ".join(x_columns)}' # 'price ~ body_style_hardtop + body_style_hatchback + body_style_sedan + body_style_wagon'
        ols_model_results = smf.ols(formula=reg_exp, data=dataset).fit()
        results.append(('Original', ols_model_results))
        
        # Formulate the model with dummies
        if dummy_columns:
            reg_exp = f'{y_column} ~ {" + ".join(x_columns + x_columns_dummies)}' # 'price ~ body_style_hardtop + body_style_hatchback + body_style_sedan + body_style_wagon'
            ols_model_results = smf.ols(formula=reg_exp, data=dataset).fit()
            results.append(('Dummy', ols_model_results))

        # Get results
        for type, result in results:
            print(f'{type} model:')
            print(result.summary())

        # Plot if simple linear regression
        if len(x_columns) == 1:
            self.__plot_simple_linear_regression(dataset, results, y_column, x_columns[0], x_columns_dummies, dummy_columns, alpha)
        
        return results


    # Results from Linear Regression and dataset to be predicted with original model dataset at first
    def predict(self, results: list, datasets: list):
        predictions = []

        # Predict for each model (Simple and with Dummies if specified)
        for i in range(len(results)):
            type, model = results[i]
            dataset = datasets[i]
            
            prediction = model.predict(dataset)
            dataset = dataset.join(pd.DataFrame({'Predictions': prediction}))
            print(f'{type} model:')
            print(dataset)

            predictions.append(prediction)
        
        return predictions


    def regression_model(self, type: str, dataset: pd.DataFrame, y_column: str, x_columns: list, alpha: float=0, dummy_column: str=None, plot: bool=True):
        # Set the model expression
        reg_exp = f'{y_column} ~ {" + ".join(x_columns)}'
        
        # Create the selected model
        if type == 'linear':
            models = {}
            model_results = smf.ols(formula=reg_exp, data=dataset).fit()
            
            # If dummy then append Dummy model and re-create the Original model
            if dummy_column:
                models['Dummy'] = model_results
                x_columns.remove(dummy_column)

                reg_exp = f'{y_column} ~ {" + ".join(x_columns)}'
                model_results = smf.ols(formula=reg_exp, data=dataset).fit()

            # Append the original model to the models list
            models['Original'] = model_results
            
            # Plot if remains a unique x_column variable
            if plot and len(x_columns) == 1:
                self.__plot_linear_regression(dataset, y_column, x_columns[0], models, alpha, dummy_column)

        if type == 'logistic':
            model_results = smf.logit(formula=reg_exp, data=dataset).fit()
            
            # Plot if alpha is specified
            if plot:
                self.__plot_logistic_regression(dataset, y_column, x_columns[0], model_results)
        
        print(model_results.summary())