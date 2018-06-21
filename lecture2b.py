#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 00:23:27 2018

@author: enrijetashino
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas import DataFrame as df
from sklearn import datasets
import seaborn.apionly as sns

df_vt = pd.read_csv('/Users/enrijetashino/Downloads/VT_cleaned.csv', low_memory=False)

#----------------------------#
# Preview the Available Data #
#----------------------------#

df_vt.head()        # Preview the beginning of the data
    
df_vt.columns       # Show the columns' names 

#-------------------------#
# Drop the missing values #
#-------------------------#

df_vt.count()   # Count the number of observations for each column 

#--------------------------------------------------#
# Fill missing search type values with placeholder #
#--------------------------------------------------#

df_vt['search_type'].fillna('N/A', inplace = True)

#-------------------------------#
# Drop rows with missing values #
#-------------------------------#

df_vt.dropna(inplace=True)

df_vt.count()

#-----------------#
# Stops by county #
#-----------------#

df_vt['county_name'].value_counts()

#------------#
# Violations #
#------------#

df_vt['violation'].value_counts()

#------------------#
# Stops by outcome #
#------------------#

df_vt['stop_outcome'].value_counts()

#-----------------#
# Stops By Gender #
#-----------------#

df_vt['driver_gender'].value_counts()

#----------------------#
# Stops by driver race #
#----------------------#

df_vt['driver_race'].value_counts()

#---------------------------------------# 
# Police Stop Frequency by Race and Age #
#---------------------------------------#

fig, ax = plt.subplots()
ax.set_xlim(15, 70)

for race in df_vt['driver_race'].unique():
    s = df_vt[df_vt['driver_race'] == race]['driver_age']
    s.plot.kde(ax=ax, label=race)
ax.legend()

fig.savefig('/Users/enrijetashino/Downloads/plot1.png')   # save the figure to file
plt.close(fig)    # close the figure


def compute_outcome_stats(df):
    n_total = len(df)
    n_warnings = len(df[df['stop_outcome'] == 'Written Warning'])
    n_citations = len(df[df['stop_outcome'] == 'Citation'])
    n_arrests = len(df[df['stop_outcome'] == 'Arrest for Violation'])
    citations_per_warning = n_citations / n_warnings
    arrest_rate = n_arrests / n_total

    return(pd.Series(data = {
        'n_total': n_total,
        'n_warnings': n_warnings,
        'n_citations': n_citations,
        'n_arrests': n_arrests,
        'citations_per_warning': citations_per_warning,
        'arrest_rate': arrest_rate
    }))


# Test the function above 

compute_outcome_stats(df_vt)


# Breakdown By Race

df_vt.groupby('driver_race').apply(compute_outcome_stats)

# Let's visualize these results.

figsize = (16,8)

race_agg = df_vt.groupby(['driver_race']).apply(compute_outcome_stats)
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize)
race_agg['citations_per_warning'].plot.barh(ax=axes[0], figsize=figsize, title="Citation Rate By Race")
race_agg['arrest_rate'].plot.barh(ax=axes[1], figsize=figsize, title='Arrest Rate By Race')


# Create new column to represent whether the driver is white
df_vt['is_white'] = df_vt['driver_race'] == 'White'

# Remove violation with too few data points
df_vt_filtered = df_vt[~df_vt['violation'].isin(['Other (non-mapped)', 'DUI'])]


df_vt_filtered.groupby(['is_white','violation']).apply(compute_outcome_stats)
race_stats = df_vt_filtered.groupby(['violation', 'driver_race']).apply(compute_outcome_stats).unstack()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
race_stats.plot.bar(y='arrest_rate', ax=axes[0], title='Arrest Rate By Race and Violation')
race_stats.plot.bar(y='citations_per_warning', ax=axes[1], title='Citations Per Warning By Race and Violation')


# FILTERING
# Also for matrices: possible to select only those elements that meet a certain 
# condition
# But: Requires keeping track of rows and columns!
# To show how filtering works for matrices, load USArrests, a practice dataset

# US Arrests in all U.S. states

# Set ipython's max row display
pd.set_option('display.max_row', 100)

USArrests = pd.read_csv('/Users/enrijetashino/Downloads/USArrests.csv', low_memory=False)

data = USArrests
data.head()
data.columns        # Shows the column names 


# Show the first five observations by the position 

data[:5]        # Does not include the fifth 
data.loc[:5]    # It includes the fifth 
data.iloc[4]    # It returns the 4th row in the data

# To show the states with more than 250 assault cases:
data[data['Assault'] > 250]
data.loc[data['Assault'] > 250]

# Rename the first column to 'State'

data.rename(columns={'Unnamed: 0': 'State'}, inplace=True)

# We use inplace = True in order not to reassign data again

# To show the state and assault more than 250: 

data[(data['Assault'] > 250) & (data['UrbanPop'] > 60)]


data[data.columns[0]]       # By using column position 

# To select just states with a large urban population:

data[data['UrbanPop'] >= 80]

# Also multiple conditions possible:

data[(data['Assault'] > 250) & (data['UrbanPop'] >= 80)]


# Return the index of the data which satisfy certain conditions 
# Tells you the index of all observations that meet a condition:

data.index[data['Assault'] > 250].tolist()

# Remove a column from the dataframe 

data.drop(['State'], 1, inplace = True)


# Remove multiple columns in the dataframe 

data.drop(data['State', 'Assault'], 1, inplace = True)

# Convert a dataset to a matrix 
# Remove the column of state names 

data.drop(['State'], 1, inplace=True)

matdata = np.array(data)
matdata.shape


# Show the first and second column only 

data.iloc[:, 0:2]

# Using apply function in Python

data.iloc[:, 1:3].apply(np.mean, 0).tolist()
data.iloc[:, 1:3].apply(np.mean, 0)
data.iloc[:, 1:3].apply(np.sum, 0).tolist()
data.iloc[:, 1:3].apply(np.sum, 0)

# Take the sum of the first six elements in each row 

def f(x):
    y = np.sum(data.iloc[0:6,1:5])
    return(y)

f(data)

# ADDING/DELETING MATRIX ROWS AND COLUMNS
# Sometimes you want to add observations or variables to a dataset

m = np.array([1,2,4,5]).reshape(2,2)

# Add a row to the matrix created above 
# Use np.vstack([m, newList])

m = np.vstack([m, [3,6]])

# Add a new column 

m = np.hstack([m, [[1],[1],[1]]])
m.shape

# Add a row and place it in row 2 for example 
# Use np.insert()

m = np.insert(m, 1, [2,2,2], axis = 0)


# 1. FOR-LOOPS
# For-Loops repeat a certain task for multiple values
# A basic loop works like this:

for i in range(10):
    print(i)
    

# Example 1: Compute values of a function:
    
y = np.repeat(0,100)
for i in range(100):
    y[i] = i**2
    
plt.plot(y)


# Example 2: Compute changes:
# Suppose you have a time-series with GDP in each year

x = np.array([17, 17.2, 17.6, 17.0, 17.1])
dx = np.repeat(0,5)

for i in range(1,5):
    dx[i] = x[i] - x[i-1]

dx

# Example 3: To sum up all elements of a vector:

xsum = 0
for i in range(len(x)):
    xsum = xsum + x[i]
    
xsum

# 2. IF-STATEMENTS
# Often you want to execute a command only if a certain condition is met
# For example, you'll want to run a regression only if there are no "NAs" in the data
# Or: Only continue if your code ran without error
# For those cases: If-Statements:
# A basic if-statement works like this:

x = 2
if x == 2:
    print(x)


# If-Statements are very useful in loops
# E.g. to determine the index of the first 1 value in a vector:
    
x = np.array([2,6,4,1,3,9])
for i in range(len(x)):
    if x[i] == 1:
        break 
    
i
    
# break tells the loop to stop as soon it found a 1
# i is then the position of x at which the loop found 1 for the first time

# 3. WHILE-LOOPS
# While-Loops execute a task until a certain condition is met. 
# Or, put differently: It executes a command while a certain statement is true.

a = 2
while a < 100:
    a = a**2

a
