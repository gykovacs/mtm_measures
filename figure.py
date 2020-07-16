#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 15:18:32 2020

@author: gykovacs
"""

import matplotlib.pyplot as plt
import numpy as np

t= np.random.rand(20)*10
w= np.sin(t) + np.random.rand(20)/2

n_bins= 3

t_diff= (np.max(t) - np.min(t))/n_bins
t_bins= [np.min(t) + t_diff*i for i in range(1, n_bins)]
w_diff= (np.max(w) - np.min(w))/n_bins
w_bins= [np.min(w) + w_diff*i for i in range(1, n_bins)]

plt.figure(figsize=(7, 2.5))
plt.scatter(t, w)
plt.xlabel('t')
plt.ylabel('w')
plt.title('Illustration of MTM')

plt.vlines(t_bins, np.min(w), np.max(w))

t_binning= np.digitize(t, t_bins)

w_means = []
for i in np.unique(t_binning):
    w_means.append(np.mean(w[t_binning == i]))
    
for i in range(len(w_means)):
    print(i)
    if i == 0:
        bin_min= 0
    else:
        bin_min= t_bins[i-1]
    if i == len(w_means) - 1:
        bin_max= np.max(t)
    else:
        bin_max= t_bins[i]
    print(w_means[i], bin_min, bin_max)
    plt.hlines(w_means[i], bin_min, bin_max, colors='red')
plt.show()

plt.figure(figsize=(7, 2.5))
plt.scatter(t, w)
plt.xlabel('t')
plt.ylabel('w')
plt.title('Binning in MI')

plt.vlines(t_bins, np.min(w), np.max(w))
plt.hlines(w_bins, np.min(t), np.max(t))
plt.show()