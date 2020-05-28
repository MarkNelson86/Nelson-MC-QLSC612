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

import csv
import os.path
import pandas as pd
import numpy as np


## Read in CSV file to pandas DataFrame
filepath = '/Users/mheado86/Desktop/QLS612_Assessment/Nelson-MC-QLSC612/practical/'
file_name = 'brainsize'
full_fn_csv = os.path.join(filepath, file_name+'.csv')

with open(full_fn_csv) as file:
  brainsize = pd.read_csv(file,sep=";")                                         # read csv file with ; as separater
  
del brainsize['Unnamed: 0']                                                     # delete extra column generated automatically
# alternatively could use brainsize.drop('Unnamed: 0',1) ; command works for rows too


## Add random variable partY to each entry in dataframe
brainsize['partY'] = pd.Series(np.random.randn(len(brainsize)), index=brainsize.index)