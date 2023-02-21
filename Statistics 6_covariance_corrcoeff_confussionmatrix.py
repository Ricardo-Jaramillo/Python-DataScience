import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# Get a visual representation of the Covariance Matrix
A = [45, 37, 42, 35, 39]
B = [38, 31, 26, 28, 33]
C = [10, 15, 17, 21, 12]

data = np.array([A, B, C])

        # cov_matrix = pd.DataFrame.cov(df)
cov_matrix = np.cov(data, bias=False)
print(cov_matrix)
sns.heatmap(cov_matrix, annot=True, fmt='g')
plt.show()


# Correlation Coefficient
corrcoeff_matrix = np.corrcoef(data, bias=False)
print(corrcoeff_matrix)
sns.heatmap(corrcoeff_matrix, annot=True, fmt='g')
plt.show()


# Confussion Matrix
data = {'y_actual':    [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        'y_predicted': [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0]
        }

df = pd.DataFrame(data)

''' # Working with non numeric data
df['y_actual'] = df['y_actual'].map({'Yes': 1, 'No': 0})
df['y_predicted'] = df['y_predicted'].map({'Yes': 1, 'No': 0})
'''

confusion_matrix = pd.crosstab(df['y_actual'], df['y_predicted'], rownames=['Actual'], colnames=['Predicted'], margins=True)
print(confusion_matrix)

sns.heatmap(confusion_matrix, annot=True)
plt.show()
