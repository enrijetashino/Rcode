#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 02:16:27 2018

@author: enrijetashino
"""

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
# A 6th Session on Python #
#-------------------------#

CarData = pd.read_csv("/Users/enrijetashino/Desktop/Python Lectures/fueleconomy.csv")
CarData = pd.DataFrame(CarData)

#-------#
# Plots #
#-------#

# With circle blue colors.
plt.figure(1) 
plt.plot(CarData['hwy'], CarData['cty'], 'bo')
plt.xlabel('hwy')
plt.ylabel('cty')
plt.title('Scatter plot of hwy vs cty')
plt.grid(True)
plt.show()

# With red crosses.
plt.figure(2) 
plt.plot(CarData['hwy'], CarData['cty'], 'r+')
plt.xlabel('hwy')
plt.ylabel('cty')
plt.title('Scatter plot of hwy vs cty')
plt.grid(True)
plt.show()

# You can also change the size of the points depending on a variable
# E.g. if you want to display more common brands in terms of observations 
# in a larger way, use "size":
CarData['counts'] = CarData.groupby(['make'])['make'].transform('count')

plt.figure(3)
plt.scatter(CarData['hwy'], CarData['cty'], marker='o', c='r', s=CarData['counts'])
plt.xlabel('hwy')
plt.ylabel('cty')
plt.title('Scatter plot of hwy vs cty')
plt.grid(True)
plt.show()


# We use sns.regplot or sns.lmplot also  
sns.regplot(x='cty', y='hwy', marker="+", ci=95, data=CarData)
sns.lmplot(x="cty", y="hwy", marker="o", ci=95, data=CarData)



