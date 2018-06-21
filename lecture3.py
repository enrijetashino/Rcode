#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 16:26:22 2018

@author: enrijetashino
"""

#------------------------#
# A 3rd Python Session   #
#------------------------#


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas import DataFrame as df
from sklearn import datasets
import seaborn.apionly as sns



# Create a dataframe 

dt = {'kids': ['Jack', 'Jill', 'John'], 
      'ages': [12, 10, 11], 
      'height': ["4'10", "4'5", "4'8"]}

dt = pd.DataFrame(data = dt)

new = [1, 2, 3]
dt['new'] = new         # Add a new column to dataframe

# Renamme the column new 
dt.rename(columns = {'new': 'id'}, inplace = True)

# Create a new dataset 
dt2 = {'kids': ["Noah", "Emma"], 'ages': [8,10], 'height': ["4'3", "4'5"], 
       'id': [4, 5]}

dt2 = pd.DataFrame(dt2)

dt = dt.append(dt2, ignore_index=True)
# Or you can ignore the index 

# ACCESSING DATA FRAMES IN PYTHON

# To select the variable "kids", use []
dt['kids']

# Or we can use iloc[]

dt.iloc[:,3]

# To select certain observations, use []:

dt.iloc[0,:]
dt.iloc[2:4,:]

# Import babynames dataset

dt = pd.read_csv("/Users/enrijetashino/Desktop/babynames.csv")
dt = pd.DataFrame(dt)

dt.head(5)       # Shows the first observations only
dt.tail(5)       # Shows the last observations only


dt.columns      # Column names
len(dt)         # Number of rows 

ones = np.repeat(1,len(dt))
ones = pd.Series(ones)

dt = dt.assign(constant = ones.values)      # Add a new column of ones

dt['constant']

# Change the name of the new first and last column

dt.rename(columns = {'Unnamed: 0': 'obs', 'constant': 'ones'}, inplace = True)

# Describe the data

dt.describe()


# First, let's look at the most common girl names
# The data is sorted such that it shows the most popular girl name in each year first
# -> Want to keep only the first observation (row) for each year:

dt.groupby('year').first()

# Now, let's get the most common boy names
# First, let's restrict the sample to boys:

dt_boys = dt[dt.sex == "M"]

dt_boys.groupby(['year']).first()

# Now, suppose we want both

popular_names = dt.groupby(["year", "sex"]).first()

# Very often you'll want to sort data in certain way
# For example, in the popular_names - file you might want to show girls first and then
# boys:

# popular_names.sort_values('sex')

data = pd.read_csv("/Users/enrijetashino/Desktop/fueleconomy.csv")

# Dataset shows miles per gallon for a broad range of cars in different years

# Let's get rid of Electric cars and just look at 2015

data_sub = data[(data['year'] == 2015) & (data['fuel'] != 'Electricity')]
data_sub.head(5)

# USING FUNCTIONS ON DATA FRAMES
# You can use functions both on variables and observations
# For example:

data['hwy'].mean()
data['hwy'].describe()

# You can also differentiate by group:

EffYear = data.groupby(['year'])['hwy'].mean()
EffClass = data.groupby('class')['hwy'].mean()

# You can either delete a variable with

data.drop(['trans'], 1, inplace=True)   # Delete variable trans

# Or decide which ones you want to keep

data_sub = data[['make', 'model', 'hwy']]
data_sub.head(5)

data_sub = data[list(data.columns[:2])] # Keep columns 0 and 1

# It is particularly convenient to restrict your dataset to only some observations
# Use square brackets [] to include conditions that select only a subset of observations:

data_sub = data[data["year"] >= 2000]
data_sub = data[(data["year"] >= 2000) & (data["make"] == "Ford")]
data_sub = data[(data["make"] == "Buick") | (data["make"] == "Ford")]

# To show the number of observations by make

ObsMake = data.groupby(["make"]).size()

# Or using this way

ObsMake = data.groupby(["make"])["make"].agg(['count'])

# Or it can be done this way

ObsMake = data['make'].value_counts()

# Show the number of models by make in year 1984

data[data['year'] == 1984].groupby('make')['model'].count().to_frame()

# Show the number of models by make through all the years

all_years_make = data.groupby(['year', 'make'])['model'].count().to_frame()

# You can also restrict the sample to only those brands that have at least 
# a minimum nr of observations:


# First create the variable "Counts" as follows and add it to the dataframe 

data.groupby(['make'])['make'].agg(['count'])

data['Counts'] = data.groupby(['make'])['make'].transform('count')

# Now keep only the observations for which "Counts" > 1000

data[data['Counts'] > 1000]


























