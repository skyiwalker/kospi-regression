import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

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
XL = data.iloc[:, 0].values # date column
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
plt.xlim((1,241))
plt.ylim((500,3000))
plt.scatter(XL,Y,color='red')
plt.xticks(np.arange(0,241,40))
# plt.plot(XL, f(X), label="polynomial")
# plt.plot(XL, f2, label="logarithm")
# plt.plot(XL, f2_scipy, label="log & fix y-intercept")
plt.plot(XL, f3, label="exponential")
# plt.plot(X, func(X, *popt))
plt.legend()
plt.grid()
plt.xlabel('Date')
plt.ylabel('KOSPI')
plt.show()

# Predictions

# make date table
str_date_list = []
start_date = datetime.strptime('200002', '%Y%m')
end_date = datetime.strptime('204002', '%Y%m')
while start_date.strftime('%Y%m') != end_date.strftime('%Y%m'): 
    str_date_list.append(start_date.strftime('%Y%m'))
    start_date = start_date + relativedelta(months=1)
datelen = len(str_date_list)
X2 = np.arange(1,datelen+1,1)
f3_predict = expfit[0][0] * np.exp(expfit[0][1]*X2) + expfit[0][2]
f2_predict = logfit[0]*np.log(X2) + logfit[1]
f2_sci_predict = logfit_scifit[0][0]*np.log(X2) + Y[0]

plt.figure(2)
plt.xlim((1,480))
plt.xticks(np.arange(1,datelen+1,40))
# plt.scatter(str_date_list,Y,color='red')
# plt.plot(str_date_list, f(X2), label="polynomial")
# plt.plot(str_date_list, f2_predict, label="logarithm")
# plt.plot(str_date_list, f2_sci_predict, label="log & fix y-intercept")
plt.plot(str_date_list, f3_predict, label="exponential")
plt.grid()
plt.xlabel('Date')
plt.ylabel('KOSPI')
plt.show()