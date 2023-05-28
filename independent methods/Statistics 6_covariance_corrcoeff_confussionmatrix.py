import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# Get a visual representation of the Covariance Matrix
A = [0, 3, 5, 9, 11]
B = [-2, -6, -8, -11, -13]
C = [-10, -15, -17, -21, -27]

data = np.array([A, B, C])

        # cov_matrix = pd.DataFrame.cov(df)
cov_matrix = np.cov(data)
print(cov_matrix)
sns.heatmap(cov_matrix, annot=True, fmt='g', xticklabels=['A', 'B', 'C'], yticklabels=['A', 'B', 'C'])
plt.show()


# Correlation Coefficient
corrcoeff_matrix = np.corrcoef(data, bias=False)
print(corrcoeff_matrix)
sns.heatmap(corrcoeff_matrix, annot=True, fmt='g', xticklabels=['A', 'B', 'C'], yticklabels=['A', 'B', 'C'])
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
