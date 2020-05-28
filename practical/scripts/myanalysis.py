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
from scipy import stats as sst
#import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm


## Read in CSV file to pandas DataFrame & manipulate data

# define file info
filepath = '/Users/mheado86/Desktop/QLS612_Assessment/Nelson-MC-QLSC612/practical/'
file_name = 'brainsize'
full_fn_csv = os.path.join(filepath, file_name+'.csv')

# open into dataframe
with open(full_fn_csv) as file:
  BS = pd.read_csv(file,sep=";")                                                # read csv file with ; as separater
  
del BS['Unnamed: 0']                                                            # delete extra column generated automatically
# alternatively could use BS.drop('Unnamed: 0',1) ; command works for rows too


# Add random variable partY to each entry in dataframe
#BS['partY'] = pd.Series(np.random.randn(len(BS)), index=BS.index)              # continuous random number
BS['partY'] = pd.Series(np.random.randint(0,10,size=len(BS)), index=BS.index)   # catagorical

# Prepare data by removing rows with missing values and forcing categorical & float dtype
BSmod = BS.replace('Female', 0)
BSmod = BSmod.replace('Male', 1)
miss_vals_rowi = [BSmod[BSmod['Weight'] == '.'].index.values]
BSmod = BSmod.drop(BSmod.index[miss_vals_rowi])                                 # del rows with missing values
BSmod = BSmod.astype('float')
BSmod['Gender'] = BSmod['Gender'].astype('category')                            # all types = float except gender = categorical
# BSmod['partY'] = BS['partY'].drop(BS.index[miss_vals_rowi])                   # to switch out partY vectors







## build regression model using sklearn.linear_model

# separate predictors and dependent variable
col_names = BS.columns.tolist()                                                 # extract column names as list
X = BS.drop(col_names[-1],1)                                                    # isolate predictors (rm Y values)
Y = BS[col_names[-1]]
Y = BSmod[col_names[-1]]


# prepare data for model
#BS.dtypes                                                                       # show column datatypes
#X = X.replace('Female', 0)                                                      # replace gender strings with binary
#X = X.replace('Male', 1)
#X['Gender'].astype('category')                                                  # and change to categorical type
#
#X[X == '.']                                                                     # shows location of missing values
#miss_vals_rowi = X[X['Weight'] == '.'].index.values
#X = X.drop(X.index[miss_vals_rowi])                                             # rm rows with missing values
#Y = Y.drop(Y.index[miss_vals_rowi])                                             # from Y vector also
#
#X = X.astype('float')                                                           # changes all values to float type
## alternatively could change individual column types using: X['Weight'] = X['Weight'].astype('float')

# model using sklearn.linear_model
lm = linear_model.LinearRegression()
model_skl = lm.fit(X,Y)
predictions_skl = lm.predict(X)                                                 # predicts Y from X using model

# Print out the statistics
print(predictions_skl)
lm.score(X,Y)                                                                   # R^2 score of our model
lm.coef_                                                                        # our model coeffs
lm.intercept_

# Visualize 
plt.scatter(X[col_names[0]], Y,  color='black')                                 # plot X columns vs Y individually
plt.plot(X[col_names[0]], predictions_skl, color='blue', linewidth=3)           # X columns vs predicted Y
plt.xticks(())
plt.yticks(())
plt.show()

for p in range(len(col_names)-1):                                               # How to plot multiple plots together??
    plt.scatter(X[col_names[p]], Y,  color='black')



## Regression with statsmodels

X = X.astype(float)
model_sm = sm.OLS(Y.astype(float), X.astype(float)).fit()
predictions_sm = model_sm.predict(X)                                            # make the predictions by the model
model_sm.summary()                                                              # Print out the statistics





## HYPOTHESIS TESTING WITH SCIPY & SEABORN

# 2-sample t-test: male vs female populations
female_partY = BS[BS['Gender'] == 'Female']['partY']
male_partY = BS[BS['Gender'] == 'Male']['partY']
male_vs_female_partY_tstat = sst.ttest_ind(female_partY, male_partY)[0]
male_vs_female_partY_pval = sst.ttest_ind(female_partY, male_partY)[1]

# 2-sample t-test: Columnwise, all variables, split by + or - partY value: p-hacking
BSmod_partY_neg = BSmod[BSmod['partY'] < 0]
BSmod_partY_pos = BSmod[BSmod['partY'] > 0]
stats_store_sign = []                                                           # store results of tests in nested list

for col in range(len(col_names)):                                               # loop through columns
    BSmod_partY_neg_tempvals = BSmod_partY_neg[col_names[col]]                  # get values for tests
    BSmod_partY_pos_tempvals = BSmod_partY_pos[col_names[col]]
    stats_store_sign.append(sst.ttest_ind(BSmod_partY_neg_tempvals, BSmod_partY_pos_tempvals)) # tests
    
for i in range(len(stats_store_sign)):                                               # print p values
    print(stats_store_sign[i][1])
    
    
# 2-sample t-test: large or small partY vs other variables: p-hacking
BSmod_partY_small = BSmod[abs(BSmod['partY']) < .5]
BSmod_partY_large = BSmod[abs(BSmod['partY']) >= .5]
stats_store_size = []                                                           # store results of tests in nested list

for col in range(len(col_names)):                                               # loop through columns
    BSmod_partY_small_tempvals = BSmod_partY_small[col_names[col]]              # get values for tests
    BSmod_partY_large_tempvals = BSmod_partY_large[col_names[col]]
    stats_store_size.append(sst.ttest_ind(BSmod_partY_small_tempvals, BSmod_partY_large_tempvals)) # tests
    
for i in range(len(stats_store_size)):                                               # print p values
    print(stats_store_size[i][1])
    

# 2-sample t-test for binary Y: 0 or 1 partY vs other variables: p-hacking
BSmod_partY_0 = BSmod[BSmod['partY'] == 0]
BSmod_partY_1 = BSmod[BSmod['partY'] == 1]
stats_store_bin = []                                                            # store results of tests in nested list

for col in range(len(col_names)):                                               # loop through columns
    BSmod_partY_0_tempvals = BSmod_partY_0[col_names[col]]                      # get values for tests
    BSmod_partY_1_tempvals = BSmod_partY_1[col_names[col]]
    stats_store_bin.append(sst.ttest_ind(BSmod_partY_0_tempvals, BSmod_partY_1_tempvals)) # tests
    
for i in range(len(stats_store_bin)):                                               # print p values
    print(stats_store_bin[i][1])
    



## Definite P-hacking: FIND THE SIGNIFICANT VALUES!!!

# 2-sample t-test: columnwise, by categorical partY value vs remainder, for each unique partY value individually
BSmod_partY_unique = BSmod.partY.unique()
BSmod_partY_unique.sort()
#BSmod_partY_0 = BSmod[BSmod['partY'] == 0]
#BSmod_partY_1 = BSmod[BSmod['partY'] == 1]
stats_store_cat_out = []                                                        # store results of tests in nested list
WINNERS = {}

for val in range(len(BSmod_partY_unique)):                                      # iterate through categorical Y values
    BSmod_partY_valu = BSmod[BSmod['partY'] == BSmod_partY_unique[val]]
    BSmod_partY_rest = BSmod[BSmod['partY'] != BSmod_partY_unique[val]]
    stats_store_cat_in = []
    
    for col in range(len(col_names)-1):                                         # loop through columns
        BSmod_partY_valu_tempvals = BSmod_partY_valu[col_names[col]]            # get values for tests
        BSmod_partY_rest_tempvals = BSmod_partY_rest[col_names[col]]
        
        t, p = sst.ttest_ind(BSmod_partY_valu_tempvals, BSmod_partY_rest_tempvals)
        stats_store_cat_in.append([t,p])
        
        if p <= .05:
            WINNERS['y_value'] = BSmod_partY_unique[val]
            WINNERS['x_column'] = col_names[col]                                # Y = 9, X = 'MRI_Count'
        
    stats_store_cat_out.append(stats_store_cat_in)
        
for i in range(len(stats_store_cat_out)):                                       # print p values
    
    for j in range(len(stats_store_cat_out[i])):
        print(stats_store_cat_out[i][j][1])
    

## Verify WINNERS

BSmod_partY_9 = BSmod[BSmod['partY'] == 9]
BSmod_partY_rest = BSmod[BSmod['partY'] != 9]

BSmod_partY_9_tempvals = BSmod_partY_9['MRI_Count']                      # get values for tests
BSmod_partY_rest_tempvals = BSmod_partY_rest['MRI_Count']

t, p = sst.ttest_ind(BSmod_partY_9_tempvals, BSmod_partY_rest_tempvals)         # p < .039!!!



## Visualize significant result!
    


## Pairwise correlations








