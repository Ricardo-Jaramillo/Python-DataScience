import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pareto

def my_pareto_pdf(x, a, x_m):
    """
    Returns the value of the pareto density function at
    the point x.
    """
    if x >= x_m:
        pdv = a
        pdv *= x_m**a
        pdv /= x**(a+1)
        return pdv
    else:
        return 0

x = np.linspace(0, 10, 100)

plt.plot(x, pareto.pdf(x, b=1.3), color='k', label='Scipy: b=1.3')
plt.plot(x, [my_pareto_pdf(val, a=1.3, x_m=1) for val in x], color='tab:blue', alpha=0.5, lw=5, label='Mypdf: a=1.3 x_m=1')

plt.plot(x, pareto.pdf(x, b=1.7, scale=3), color='k', label='Scipy: b=1.7 scale=3')
plt.plot(x, [my_pareto_pdf(val, a=1.7, x_m=3) for val in x], color='tab:blue', alpha=0.5, lw=5, label='Mypdf: a=1.7 x_m=3')

plt.plot(x, pareto.pdf(x, b=2.3, scale=6), color='k', label='Scipy: b=2.3 scale=6')
plt.plot(x, [my_pareto_pdf(val, a=2.3, x_m=6) for val in x], color='tab:blue', alpha=0.5, lw=5, label='Mypdf: a=2.3 x_m=6')

plt.legend(loc='best')
plt.title('Pareto PDFs')
plt.show()