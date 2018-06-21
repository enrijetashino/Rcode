#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 18:24:43 2018

@author: enrijetashino
"""

# pip install linearmodels in your command line

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas import DataFrame as df
from sklearn import datasets
import seaborn.apionly as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.gmm import IV2SLS
from statsmodels.sandbox.regression.gmm import IV2SLS, IVGMM, DistQuantilesGMM, spec_hausman
from statsmodels.sandbox.regression import gmm
from linearmodels.iv import IV2SLS

#-------------------------#
# A 5th Session on Python #
#-------------------------#

#-----------#
# Bootstrap #
#-----------#

# Import the dataset "Guns"
Guns = pd.read_csv("/Users/enrijetashino/Desktop/Python Lectures/Guns.csv")
Guns = pd.DataFrame(Guns)
Guns.head(5)
Guns.shape

# Consider again the regression "violent crime" on "law":
reg = smf.ols('violent ~ law', data = Guns).fit()       # No need to use C(law)
print(reg.summary())

reg.params

# Get the robust standard errors 
reg_robust = reg.get_robustcov_results()
print(reg_robust.summary())

# Now let's compute standard errors using bootstrap:

#-----------------------------------------------------------------------------#
# Bootstrap code in Python

N = 1001                                                         # Number of iterations
coeffs_bt = np.array(np.repeat(0,2*N)).reshape(N,2) 

for i in range(N):
    n = len(Guns)                                               # Number of observations in the dataset 
    dataGuns_bt = Guns.sample(n = n, replace = True)            # Resample with replacement 
    reg_bt = smf.ols('violent ~ law', data = dataGuns_bt).fit() # Run regression for each sample 
    coeffs_bt[i,:] = np.array(reg_bt.params)                    # Save the estimated parameters

#-----------------------------------------------------------------------------#

# To get the final estimates of the coefficients and standard errors, use: 
beta_bt = coeffs_bt.mean(axis=0)    # Coefficients 
beta_bt

np.cov(coeffs_bt.T)                   # Covariance Matrix (Diagonal elements is the variance of the coefficients)

se = np.sqrt(np.diag(np.cov(coeffs_bt.T)))
se

# You can also use the bootstrap method to construct confidence intervals
# These intervals are constructed such that the true parameter lies within the 
# interval with a certain probability alpha:

# For example, if alpha is 5%
coeffs_bt = pd.DataFrame(coeffs_bt)
lower = coeffs_bt.loc[:,1].quantile(0.025)
upper = coeffs_bt.loc[:,1].quantile(0.975)
CI5 = np.array([lower, upper])
    # True Value lies within this interval with 95% probability  


# Or if 1%
lower = coeffs_bt.loc[:,1].quantile(0.005)
upper = coeffs_bt.loc[:,1].quantile(0.995)
CI1 = np.array([lower, upper])
    # True Value lies within this interval with 99% probability 

CollegeDistance = pd.read_csv("/Users/enrijetashino/Desktop/Python Lectures/CollegeDistance.csv")
CollegeDistance = pd.DataFrame(CollegeDistance)

# Basic Regression

reg_ols = smf.ols('wage ~ education', data = CollegeDistance).fit()
print(reg_ols.summary())

# Or with controls

reg_ols = smf.ols('wage ~ education + gender + ethnicity + urban', data = CollegeDistance).fit()
print(reg_ols.summary())

# -> Surprisingly: No effect of education on wage!

# Problem: Education is not a truly independent variable, e.g. because:
# 1. More skilled people likely to get a college degree and earn higher wages either way
# 2. Family connections might help someone to get into college as well as a good job

# Controlling for those factors hard because ability or family connections are very hard to measure
# => Want an instrument!

# As a solution, David Card used the distance to the nearest 4-year college as an instrument for education:
# - Affects probability of going to college
# - Does typically not directly affect eventual wages

# Dataset on high school students
# In 1980: Were asked about their distance to a four-year college -> Variable "distance"
# In 1986: Were asked about their years of education -> Variable "education"

# Were kids that lived closer to a college more likely to attend one?
CollegeDistance.plot(x = 'distance', y = 'education', kind='scatter')
plt.show()

# Or it can be plotted another way
plt.scatter(CollegeDistance['distance'], CollegeDistance['education'])

# Run the regression
reg = smf.ols('education ~ distance', data = CollegeDistance).fit()
print(reg.summary())

# Still true with controls?
reg = smf.ols('education ~ distance + gender + ethnicity + unemp + urban', data = CollegeDistance).fit()
print(reg.summary())

# And robust standard errors 
reg_robust = reg.get_robustcov_results(cov_type='HC1')
print(reg_robust.summary())

# Ok, so let's use it as an instrument
# To run an Instrumental Variables Regression, use the command IV2SLS:

CollegeDistance['const'] = 1

iv = IV2SLS(dependent=CollegeDistance['wage'],
            exog=CollegeDistance[['const', 'gender', 'ethnicity', 'unemp','urban']],
            endog=CollegeDistance['education'],
            instruments=CollegeDistance['distance']).fit()

print(iv.summary)




