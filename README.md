# Python for Data Science

## Requirements
### Libraries
- matplotlib
- scipy
- seaborn
- pandas
- numpy
- pyodc (if you need to connect to a database)

### Dataset
For this project I used a connection to a SQL Database, so the pyodbc library is installed. But you could either import a csv file or make a connection to where your dataset is located.


## Statistics Class
### Plots
1. Pareto's Chart
2. Histogram
3. Bars' charts
4. Scatter Plot
5. Probability Plot

### Measures of central tendency (and other statistical metrics)
6. Skew
7. Variance, Covariance, Standard Deviation and Variance coefficient
8. Correlation coefficient
9. Standarize Normal Distribution (zscore)
10. Central Limit Theorem
11. Confidence Intervals. Z & T for var population known or unknown, respectively
12. Confidence Interval. Dependent Samples
13. Confidence Interval. Independent Samples
    1. Var populations known (populations normally distributed, var populations known, sample sizes differ)
    2. Var populations unknown but assumed are equals
    3. Var populations unknown but assumed are different
14. Hypothesis Test. (Applied for each type of Confidence Interval)

## Regressions Class
1. Simple linear Regression (OLS)
2. Multiple linear Regression (OLS)
    1. Dummy variables
3. Logistic Regression

## Pending
1. Probability distributions
2. Plot Confidence Intervals in normal distributions and probplots
3. Simple linear Regression
    1. Other solution methods
    2. Log transformation variables as elasticity (economics)
    3. Durbin-Watson to detect no autocorrelation
    4. Other regression models to time-series or when the error terms (No Autocorrelation) are correlated
4. Logistic Regression
    1. Code
    2. Params significance on a difference of possibilities. How odds increment
    3. Binary predictors (similar to dummy variables)
    4. Caluclate the accuracy of the model
    5. Test the model accuracy. Separate Trainning and Test Samples