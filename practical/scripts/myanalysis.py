#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  This script performs operations outined in the end of week assessment in QLS612
a part of the Brain Hack School 2020 virtual summer course. The goals of this
script are to:
    
    (1) Read in data file brainsize.csv
    (2) add random noise variable partY & add to file
    (3) Generate significant (P<0.05) associations between partY & other variables in 
        brainsize.csv (correlations, predictions, etc).
        - Include results of significance tests & at least 1 figure
    (4) Generate 2nd random variable, partyY2, with different random seed and rerun same 
        association/prediction models.
        - Include in results
    

Created on Thu May 28 07:36:05 2020

@author: mheado86
"""

# import csv
import os.path
import pandas as pd
import numpy as np
from sklearn import linear_model
#  import statsmodels.api as sm


## Read in CSV file to pandas DataFrame

# define file info
filepath = '/Users/mheado86/Desktop/QLS612_Assessment/Nelson-MC-QLSC612/practical/'
file_name = 'brainsize'
full_fn_csv = os.path.join(filepath, file_name+'.csv')

# open into dataframe
with open(full_fn_csv) as file:
  brainsize = pd.read_csv(file,sep=";")                                         # read csv file with ; as separater
  
del brainsize['Unnamed: 0']                                                     # delete extra column generated automatically
# alternatively could use brainsize.drop('Unnamed: 0',1) ; command works for rows too


## Add random variable partY to each entry in dataframe

brainsize['partY'] = pd.Series(np.random.randn(len(brainsize)), index=brainsize.index)


## build regression model using sklearn.linear_model

# separate predictors and dependent variable
col_names = brainsize.columns.tolist()                                          # extract column names as list
X = brainsize.drop(col_names[-1],1)                                             # isolate predictors (rm Y values)
Y = brainsize[col_names[-1]]

# prepare data for model
brainsize.dtypes                                                                # show column datatypes
X = X.replace('Female', 0)                                                      # replace gender strings with binary
X = X.replace('Male', 1)
X['Gender'].astype('category')                                                  # and change to categorical type

X[X == '.']                                                                     # shows location of missing values
miss_vals_rowi = X[X['Weight'] == '.'].index.values
X = X.drop(X.index[miss_vals_rowi])                                             # rm rows with missing values
Y = Y.drop(Y.index[miss_vals_rowi])                                             # from Y vector also

X = X.astype('float')                                                           # changes all values to float type
# alternatively could change individual column types using: X['Weight'] = X['Weight'].astype('float')

# model using sklearn.linear_model
lm = linear_model.LinearRegression()
model = lm.fit(X,Y)
predictions = lm.predict(X)                                                     # predicts Y from X using model

# Print out the statistics
print(predictions)
lm.score(X,Y)                                                                   # R^2 score of our model
lm.coef_                                                                        # our model coeffs
lm.intercept_