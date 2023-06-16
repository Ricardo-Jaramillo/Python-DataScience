import numpy as np
from scipy import stats
from scipy.stats import norm
from numpy.random import default_rng
from numpy.random import SeedSequence

'''
loc = mean
scale = std
shape = df
df = n
size = n
sem = standard error
'''

# Setting a seed for random numbers
# print(SeedSequence().entropy)
rng = default_rng(122708692400277160069775657973126599887)

#Critical value
'''
confidence_level = 0.95
n = 10
alpha = 1 - confidence_level
a = stats.t.isf(alpha / 2, n - 1)
print(a)
'''

# Confidence Interval
'''
gfg_data = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 
            3, 4, 4, 5, 5, 5, 6, 7, 8, 10]
  
a = stats.t.interval(alpha=0.9, df=len(gfg_data)-1, loc=np.mean(gfg_data), scale=stats.sem(gfg_data))
print(a)
'''

# Descriptive Analysis compare to its theoretical distribution
'''
x = stats.t.rvs(df=10, size=1000, scale=1, loc=0, random_state=rng)

m, v, s, k = stats.t.stats(df=10, scale=1, loc=0, moments='mvsk')
n, (smin, smax), sm, sv, ss, sk = stats.describe(x)

sstr = '%-14s mean = %6.4f, variance = %6.4f, skew = %6.4f, kurtosis = %6.4f'
print(sstr % ('distribution:', m, v, s ,k))
print(sstr % ('sample:', sm, sv, ss, sk))
'''

# Standarize distribution (Central limit Theorem)
'''
data = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 
            3, 4, 4, 5, 5, 5, 6, 7, 8, 10]
data = stats.zscore(data)
print(data)
'''