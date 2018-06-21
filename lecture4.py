#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 23:33:24 2018

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


#-------------------------#
# A 4th Session on Python #
#-------------------------#

fueleconomy = pd.read_csv("/Users/enrijetashino/Desktop/Python Lectures/fueleconomy.csv")
data = pd.DataFrame(fueleconomy)

# From last time: 
#Deleting more than one variable
data.drop(['trans'], 1, inplace=True)
data.drop(['hwy', 'cty'], 1, inplace=True)

# Column and row names in data tables:
data.rename(columns = {'cyl': 'cylinders'}, inplace=True)
data.rename(columns = {'id': 'a', 'make': 'b', 'model': 'c'}, inplace=True)

# Another simple way of change the columns names is: 
data.columns.values[7] = 'cylinders'

# Change row names 

#------------------#
# Merging datasets #
#------------------#


#####################################
# Example: Creating a trade dataset #
#####################################


dt = pd.read_csv("/Users/enrijetashino/Desktop/Python Lectures/trade_data_2014.csv")
dt = pd.DataFrame(dt)

# Delete when partner is "World"
dt = dt[dt['partner'] != 'World']

# Show only imports:
dt[dt['trade_flow'] == 'Import']
dt[(dt['trade_flow'] == 'Import') & (dt['trade_value'] > 10000)]

# Another way 
dt[dt.trade_flow == 'Import']
dt[(dt.trade_flow == 'Import') & (dt.trade_value > 10000)]


# Now let's add observations for year 2015:
dt_2015 = pd.read_csv("/Users/enrijetashino/Desktop/Python Lectures/trade_data_2015.csv")
dt_2015 = pd.DataFrame(dt_2015)

dt = dt.append(dt_2015)

# And clean the dataset a little
dt.drop(['trade_flow_code', 'reporter'], 1, inplace=True)
dt = dt[dt['trade_flow'] == 'Import']

# Now suppose you have a dataset on distance between the US and other countries

distance = pd.read_csv("/Users/enrijetashino/Desktop/Python Lectures/distance-2.csv")
distance = pd.DataFrame(distance)

# To merge both datasets, using the exporting country as identifier:
dt_trade = pd.merge(dt, distance, on='partner', how='inner')
len(dt_trade)

# You can still delete observations with missing values by using:
dt_trade = dt_trade.dropna()

# Shows which rows have na. 
dt_trade[dt_trade.isnull().any(axis=1)]

#-----------------------#
# Regressions in Python #
#-----------------------#

# Import the gdp dataframe 
gdp = pd.read_csv("/Users/enrijetashino/Desktop/Python Lectures/gdp_new.csv")
gdp = pd.DataFrame(gdp)

# Now merge "gdp" dataset to the other "dt_trade" by "partner"

dt_trade = pd.merge(dt_trade, gdp, on='partner')

# Run the gravity equation

# First add a vector of ones to the dataframe "dt_trade"
dt_trade['constant'] = 1

gravity = sm.OLS(endog=dt_trade['trade_value'], exog=dt_trade[['constant', 'gdp', 'dist']], 
                 missing='drop') 
type(gravity)

results = gravity.fit()
print(results.summary())

# Another way to run regression is to use a different package
# This is quite similar to R and better than the other. 
gravity = smf.ols('np.log(trade_value) ~ np.log(gdp) + np.log(dist)', data = dt_trade).fit()
print(gravity.summary())

# Robust standard errors 
gravity_robust = gravity.get_robustcov_results()
print(gravity_robust.summary())

# Output to Latex
gravity_robust_latex = gravity_robust.summary().as_latex()
gravity_robust_latex

# Save to disc 
with open("regression_table.tex", "w") as text_file:
    text_file.write(gravity_robust_latex)

# Returns the coefficients 
print(gravity_robust.params)

# Returns confidence intervals 
gravity_robust.conf_int(alpha=0.05, cols=None)

# Returns the p-values of the coefficients 
gravity_robust.pvalues

# Returns the t-values of the coefficients 
gravity_robust.tvalues

# Returns the fitted values
fitted_values = gravity_robust.fittedvalues

# Returns the residuals 
residuals = gravity_robust.resid

# It also offers ways to adjust your regression
# Remove the intercept 

gravity_noconst = smf.ols('np.log(trade_value) ~ 0 + np.log(gdp) + np.log(dist)', data = dt_trade).fit()
print(gravity_noconst.summary())


# To use just a subset of the data
gravity = smf.ols('np.log(trade_value) ~ np.log(dist) + np.log(gdp)', data = dt_trade[dt_trade['gdp'] > 2]).fit()
print(gravity.get_robustcov_results().summary())

# And to weight observations (weight them by "gdp")
weight_gdp = np.array(dt_trade.gdp)
gravity_wls = smf.wls('np.log(trade_value) ~ np.log(dist) + np.log(gdp)', weights=weight_gdp, data = dt_trade).fit()
print(gravity_wls.summary())

# Export to latex

gravity_wls_tex = gravity_wls.summary().as_latex()
gravity_wls_tex