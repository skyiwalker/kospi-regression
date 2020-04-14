import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

# def func(x, a, b, c):
#   #return a * np.exp(-b * x) + c
#   return a * np.log(b * x) + c

data = pd.read_csv('month-kospi.csv')
# X = data.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
# X = data.index.values.reshape(-1,1)
# Y = data.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column

X = data.index.values
XL = data.iloc[:, 0].values
X = X + 1 # index start from 1
Y = data.iloc[:, 1].values

# 2nd order polynomial
pf = np.polyfit(X,Y,2)
f = np.poly1d(pf)

# logarithm function
logfit = np.polyfit(np.log(X), Y, 1)
f2 = logfit[0]*np.log(X) + logfit[1]

# curve fitting
# popt, pcov = curve_fit(func, X, Y)
logfit_scifit = curve_fit(lambda t,a: a*np.log(t)+Y[0],  X,  Y)
f2_scipy = logfit_scifit[0][0]*np.log(X) + Y[0]

expfit = curve_fit(lambda t,a,b,c: a*np.exp(b*t)+c,  X,  Y,  p0=(-10, 0.01, 100))
f3 = expfit[0][0] * np.exp(expfit[0][1]*X) + expfit[0][2]

# Fitting Curves
plt.figure(1)
plt.xlim((1,250))
plt.ylim((500,3000))
plt.scatter(X,Y,color='red')
plt.plot(X, f(X))
plt.plot(X, f2)
# plt.plot(X, func(X, *popt))
plt.plot(X, f2_scipy)
plt.plot(X, f3)
plt.show()

# Predictions
plt.figure(2)
plt.ylim((500,3000))
X2 = np.arange(0,400,1)
plt.plot(X2, f(X2))
f2 = logfit[0]*np.log(X2) + logfit[1]
f2_sci = logfit_scifit[0][0]*np.log(X2) + Y[0]
f3 = expfit[0][0] * np.exp(expfit[0][1]*X2) + expfit[0][2]
plt.plot(X2, f2)
plt.plot(X2, f2_sci)
plt.plot(X2, f3)
plt.show()