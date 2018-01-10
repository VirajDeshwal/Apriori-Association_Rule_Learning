#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 02:07:51 2018

@author: virajdeshwal
"""

#APRIORI


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

'''We will use two loops here
    1. for all transcations
    2. for all the times in a transaction.'''
    
    
transactions = []
'''As our dataset includes [7500,20] 7500 rows and 20 columns.
    so, i will be 7500(rows) and j will be 20(columns)'''
for i in range(1,7501):
    transactions.append([str(file.values[i,j]) for j in range(0,20)])
    
#Let's train the Apriori on the given dataset.
from apyori import apriori

#Let's define the rules for the Apriori 
'''min_support, min_confidence, min_lift are tuneable values. Set them to 
    minimum according to the given dataset.'''
rules =apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)

#visualizing the results

results = list(rules)
print(results)
