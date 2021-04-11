#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 19:40:38 2021

@author: hrobjarturhoskuldsson
"""

import numpy as np
import math

def expected_value(samples):
    E_X_summ = 0
    E_log_X_summ = 0
    E_neglog_P_X_summ = 0
    
    for x in samples:
        E_X_summ  += x
        E_log_X_summ += math.log(x, 10)
        E_neglog_P_X_summ += (-math.log(x, 10) * ((1/2) * math.exp(-x/2)))
    
    E_X = E_X_summ/N
    E_log_X = E_log_X_summ/N
    E_neglog_P_X = E_neglog_P_X_summ/N
    
    return E_X[0], E_log_X, E_neglog_P_X

# Takes a single uniformly distributed sample, returns the result of passing it through inverse of PX(x)
def inverse_Px(x):
    return -2 * np.log(2 * x)

# Takes amount of samples as parameter, returns samples from the exponential PDF
def get_samples(N):
    samples = []
    
    while len(samples) < N:
        s = np.random.uniform(0,1,1)
        result = inverse_Px(s)
        if result > 0:
            samples.append(result)
    
    return samples


if __name__ == "__main__":
    N = [5,10,100]
    
    for N in N:
        samples = get_samples(N)
        E_X, E_log_X , E_neglog_P_X = expected_value(samples)
        
        print('\nAmount of samples : ',len(samples))
        print('Estimated mean :', E_X)
        print('Estimated log_mean :', E_log_X)
        print('Estimated neglog_P_X :', E_neglog_P_X)






