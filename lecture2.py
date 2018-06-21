#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 22:56:23 2018

@author: enrijetashino
"""

#------------------------#
# A 2nd Python Session   #
#------------------------#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas import DataFrame as df
from sklearn import datasets
import seaborn.apionly as sns

#-----------#
# Functions #
#-----------#

#1. Add the number 1 to a vector 

def PlusOne(x):
    y = x + 1
    return(y)

x = np.array([3,5,2])
PlusOne(x)

# Functions can have arbitrarily many arguments, e.g.:
# 2. Add any number c to a vector

def PlusC(x,c):
    z = x + c 
    return(z)

PlusC(x,3)

#3. Summing up vectors x and y:

def SumVectors(x,y):
    z = x + y
    return(z)

SumVectors(x, [2,3,4])

# The output of a function can be of any mode, e.g.:
# 4.Test if the first elements of 2 vectors are the same

def EqualFirst(x,y):
    z = x[0] == y[0]
    return(z)

EqualFirst(np.array([3,5]), np.array([3,8]))
EqualFirst(np.array([4,5]), np.array([3,8])) 
    
#--------------------#
# Matrices in Python #
#--------------------#

# Create a 2x2 matrix 

x = np.array([1,2,3,4]).reshape(2,2)
x

# You can also specify elements individually:

z = np.zeros(4).reshape(2,2)

z[0,0] = 1
z[0,1] = 2
z[1,0] = 3
z[1,1] = 4

# FILTERING
# Also for matrices: possible to select only those elements that meet a certain 
# condition
# But: Requires keeping track of rows and columns!
# To show how filtering works for matrices, load USArrests, a practice dataset

from sklearn.datasets import load_iris
iris = load_iris()
column_names = iris.feature_names

df = pd.DataFrame(iris.data, column_names)


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