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

## Read in CSV file