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
        pass


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
                data_copy[column] = dataset[column].map({'Yes': 1, 'No': 0})

        return data_copy
    

    def __plot_simple_regression(self, dataset, model_results, y_column, x_column, x_column_dummies, alpha):
        # Plot x, y
        y = dataset[y_column]
        x = dataset[x_column]
        
        # fig, ax = plt.subplots()
        fig = plt.scatter(x, y)
        
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
                fig = plt.plot(x, yhat, label='Regression line')

                # Plot confidence bounds
                if alpha:
                    prstd, iv_l, iv_u = wls_prediction_std(model, alpha=alpha)
                    fig = plt.plot(x, iv_u, 'r--')
                    fig = plt.plot(x, iv_l, 'r--')

            # Dummy_model Regression Line Equations
            if model_type == 'Dummy':
                # Set Regression Line for each dummy variable
                for column in dataset[x_column_dummies]:
                    yhat_dummies = params[0] + params[column] + params[x_column]*dataset[x_column]
                    
                    # Plot each dummy Regression Line
                    fig = plt.plot(x, yhat_dummies, label=column)
            
        plt.legend(loc='best')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.show()
        
        return
        # Get params from models
        original_model_params = model_results[0].params
        original_model_rsquared = model_results[0].rsquared

        dummy_model_params = model_results[1].params
        dummy_model_rsquared = model_results[1].rsquared
        
        # Original Regression Line Equation
        yhat = original_model_params[0] + original_model_params[x_column]*dataset[x_column]

        # Dummy_model Regression Line Equation
        yhat_dummies = []
        for column in dataset[x_column_dummies]:
            a = dummy_model_params[0] + dummy_model_params[column] + dummy_model_params[x_column]*dataset[x_column]
            yhat_dummies.append(a)

        # Set x1 with the x variable to be plot
        y = dataset[y_column]
        x = dataset[x_column]

        # fig, ax = plt.subplots()
        fig = plt.scatter(x, y, color='blue')
        
        # Set regression line equations
        fig = plt.plot(x, yhat, c='orange', label ='regression line')
        fig = plt.plot(x, yhat_dummies[0], c='green', label ='regression line')
        fig = plt.plot(x, yhat_dummies[1], c='red', label ='regression line')
        
        # Get confidence bounds
        if alpha:
            prstd, iv_l, iv_u = wls_prediction_std(model_results, alpha=alpha)
            fig = plt.plot(x, iv_u, 'r--')
            fig = plt.plot(x, iv_l, 'r--')
        
        plt.legend(loc='best')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
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
        results = []
        dataset = pd.get_dummies(data=dataset, columns=dummy_columns, dtype=float)

        for dummy in dummy_columns:
            # x_columns_plot = dataset.loc[:, [dummy not in item and y_column != item for item in dataset.columns]].columns.to_list()
            x_columns_dummies = dataset.loc[:, [dummy in item for item in dataset.columns]].columns.to_list()

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
        for _, result in results:
            print(result.summary())

        # Plot if simple linear regression
        if len(x_columns) == 1:
            self.__plot_simple_regression(dataset, results, y_column, x_columns[0], x_columns_dummies, alpha)
        
    
    # n x 3 dataset. Specify dummy column to transform/map
    def dummy_variables(self, dataset: pd.DataFrame, dummy_column: str):
        # Map dummy_column
        new_dataset = self.transform('map', dataset, [dummy_column])
