#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 14:31:24 2020

@author: gykovacs
"""

import numpy as np
import itertools

d= 10
b= 3

dummy= np.random.rand(100, d)
C= np.cov(dummy, rowvar=False)

n_u= np.random.randint(1, 3, d)
#n_u= np.repeat(1, d)
cum_n_u= np.hstack([[0], np.cumsum(n_u)])

full_d= np.sum(n_u)

S_u= np.zeros(shape=(full_d, len(n_u)))
for i in range(len(cum_n_u)-1):
    for j in range(cum_n_u[i], cum_n_u[i+1]):
        S_u[j, i]= 1.0

splits= sorted(np.random.randint(1, d, b-1))
while len(np.unique(splits)) < b-1:
    splits= sorted(np.random.randint(1, d, b-1))    
bins= np.array([0] + splits + [d])

def block_sum(i):
    s= 0.0
    for j in range(bins[i], bins[i+1]):
        for k in range(bins[i], bins[i+1]):
            s+= C[j,k]*n_u[j]*n_u[k]
    return s

def row_col_sums(i, b_j):
    # i: unique row idx
    # b_j: bin index
    s= C[i,i]*n_u[i]*n_u[i]
    for j in range(bins[b_j], bins[b_j+1]):
        if i != j:
            s+= (C[i, j] + C[j, i])*n_u[i]*n_u[j]
    return s

sums= np.repeat(0.0, b)

for i in range(b):
    sums[i]= block_sum(i)

ns= np.repeat(0.0, b)
for i in range(b):
    ns[i]= cum_n_u[bins[i+1]] - cum_n_u[bins[i]]

objective= 0.0

for i in range(b):
    objective+= sums[i]/ns[i]

S= np.zeros(shape=(full_d, b))
k= 0
for i in range(0, len(bins)-1):
    for j in range(bins[i], bins[i+1]):
        for _ in range(n_u[j]):
            S[k, i]= 1
            k= k + 1

A= np.dot(np.dot(S, np.linalg.inv(np.dot(S.T, S))), S.T)
C_mod= np.dot(np.dot(S_u, C), S_u.T)

np.sum(A*C_mod)

print(bins, ns)

def changes(i, step=-1):
    offset= int((step-1)/2)
    sum_i= sums[i] + (-1)*step*row_col_sums(bins[i]+offset, i)
    sum_im1= sums[i-1] + step*row_col_sums(bins[i]+offset, i-1)
    ns_i = (ns[i] - step*n_u[bins[i]+offset])
    ns_im1 = (ns[i-1] + step*n_u[bins[i]+offset])
    
    change= (sum_i)/ns_i - sums[i]/ns[i] + (sum_im1)/ns_im1 - sums[i-1]/ns[i-1]
    
    return change, sum_i, sum_im1, ns_i, ns_im1

def greedy_binning(maxit= 10):
    it= 0
    global objective
    while True and it < maxit:
        print(it)
        it+= 1
        
        change_obj, change_idx, step_, new_sum_i, new_sum_im1, new_ns_i, new_ns_im1= 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        for i in range(1, b):
            for step in [-1, 0]:
                if ns[i + step] > n_u[bins[i] + step]:
                    change, sum_i, sum_im1, ns_i, ns_im1 = changes(i, step*2 + 1)
                    if change > change_obj:
                        change_obj, change_idx, step_, new_sum_i, new_sum_im1, new_ns_i, new_ns_im1= change, i, step*2 + 1, sum_i, sum_im1, ns_i, ns_im1
            
        print(objective, change_obj, change_idx, step_, new_sum_i, new_sum_im1)
        
        if change_obj > 0.0:
            objective= objective + change_obj
            bins[change_idx]+= step_
            sums[change_idx]= new_sum_i
            sums[change_idx-1]= new_sum_im1
            ns[change_idx]= new_ns_i
            ns[change_idx-1]= new_ns_im1
            
            #for i in range(b):
            #    sums[i]= block_sum(i)
            
            print(bins, ns)
            print('new_obj', objective, change_obj, change_idx, step, new_sum_i, new_sum_im1, new_ns_i, new_ns_im1)
            
            S= np.zeros(shape=(full_d, b))
            k= 0
            for i in range(0, len(bins)-1):
                for j in range(bins[i], bins[i+1]):
                    for _ in range(n_u[j]):
                        S[k, i]= 1
                        k= k + 1
            
            A= np.dot(np.dot(S, np.linalg.inv(np.dot(S.T, S))), S.T)
            C_mod= np.dot(np.dot(S_u, C), S_u.T)
            
            print('reconstructed', np.sum(A*C_mod))
        else:
            break
    
    # reconstructing A
    S= np.zeros(shape=(full_d, b))
    k= 0
    for i in range(0, len(bins)-1):
        for j in range(bins[i], bins[i+1]):
            for _ in range(n_u[j]):
                S[k, i]= 1
                k= k + 1
    return S

S= greedy_binning()


S= np.zeros(shape=(full_d, b))
k= 0
for i in range(0, len(bins)-1):
    for j in range(bins[i], bins[i+1]):
        for _ in range(n_u[j]):
            S[k, i]= 1
            k= k + 1

A= np.dot(np.dot(S, np.linalg.inv(np.dot(S.T, S))), S.T)
C_mod= np.dot(np.dot(S_u, C), S_u.T)

print('reconstructed', np.sum(A*C_mod))
