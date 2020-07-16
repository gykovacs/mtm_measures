#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 19:49:52 2020

@author: gykovacs
"""

import numpy as np

# dimensionality
n_d = 8

# bins
n_bins = 3

def generate_template():
    return np.round(np.random.rand(n_d)*10, decimals=0)

def generate_eqw_binning(t):
    t_diff= (np.max(t) - np.min(t))/n_bins
    t_binning= np.digitize(t, t_bins)
    return t_binning

def generate_S_from_binning(t_binning):
    S= np.zeros(shape=(len(t_binning), len(np.unique(t_binning))))
    for i, t_ in enumerate(t_binning):
        S[i][t_]= 1
    return S

def generate_S(t):
    t_binning = generate_eqw_binning(t)
    return generate_S_from_binning(t_binning)

def generate_unique_binning(t):
    return np.digitize(t, np.unique(t) + 0.01)

def generate_S_unique(t):
    t_binning = generate_unique_binning(t)
    return generate_S_from_binning(t_binning)

def generate_A_from_S(S):
    return np.dot(np.dot(S, np.linalg.inv(np.dot(S.T, S))), S.T)

def generate_A(t):
    return generate_A_from_S(generate_S(t))

def generate_m(t):
    return np.random.rand(len(np.unique(t)))

t= generate_template()

print('t', t)

A= generate_A(t)

print('A', A)

S_u = generate_S_unique(t)

print('S_u', S_u)

m= generate_m(t)

print('m', m)

# <AS_u m, S_u m>
np.dot(np.dot(A, np.dot(S_u, m)), np.dot(S_u, m))

# <AS_u m, AS_u m>
np.dot(np.dot(A, np.dot(S_u, m)), np.dot(A, np.dot(S_u, m)))



np.dot(np.dot(A, np.dot(S_u, m)), t)
np.dot(np.dot(S_u, m), t)


np.var(t + np.dot(S_u, m))

np.var(t)

cov_m= np.outer(m, m)

t_binning = generate_unique_binning(t)

ns= []
for i in np.unique(t_binning):
    ns.append(np.sum(t_binning == i))
ns= np.array(ns)







def var(x):
    return np.mean((x - np.mean(x))**2)

var_Sm = np.var(np.dot(S_u, m))


var_Sm= np.sum(ns*(m**2))/n_d - np.dot(np.dot(ns, cov_m), ns)/(n_d**2)


var_total= np.var(t + np.dot(S_u, m))
var_t= np.var(t)

total= 0.0
for i in range(len(t)):
    for j in range(S_u.shape[0]):
        for k in range(S_u.shape[1]):
            total+= t[i]*S_u[j][k]*m[k]
total/= n_d**2
covar_minus= - 2*total

total= 0.0
for i in range(len(t)):
    for j in range(S_u.shape[1]):
        total+= t[i]*S_u[i][j]*m[j]
total/= n_d
covar_plus= 2*total

var_total

var_t + var_Sm + covar_minus + covar_plus

generate_S_from_binning(t_binning)


np.var(np.dot(S_u, m))

total= 0.0
for i in range(S_u.shape[0]):
    for j in range(S_u.shape[1]):
        for k in range(S_u.shape[0]):
            for l in range(S_u.shape[1]):
                total+=S_u[i][j]*S_u[k][l]*cov_m[j][l]
total

np.dot(np.dot(ns, cov_m), ns)



total= 0.0
for i in range(S_u.shape[0]):
    for j in range(S_u.shape[1]):
        total+= S_u[i][j]**2*m[j]**2
total

np.dot(ns, m**2)

total= 0.0
for i in range(S_u.shape[0]):
    for j in range(S_u.shape[1]):
        for k in range(S_u.shape[1]):
            total+= S_u[i][j]*S_u[i][k]*m[j]*m[k]
total


np.mean((t + np.dot(S_u, m))**2)

covar_plus + np.mean(t*t) + total


w= np.sin(t) + np.random.rand(20)/2

n_bins= 3

t_diff= (np.max(t) - np.min(t))/n_bins
t_bins= [np.min(t) + t_diff*i for i in range(1, n_bins)]
w_diff= (np.max(w) - np.min(w))/n_bins
w_bins= [np.min(w) + w_diff*i for i in range(1, n_bins)]

t_binning= np.digitize(t, t_bins)

n_bins_full= len(np.unique(t))

t_bins_full= np.unique(t) + 0.01

t_binning_full= np.digitize(t, t_bins_full)

S= np.zeros(shape=(len(t_binning), 3))

for i, t_ in enumerate(t_binning):
    S[i][t_]= 1
    
S_full= np.zeros(shape=(len(t_binning), n_bins_full))

for i, t_ in enumerate(t_binning_full):
    S_full[i][t_]= 1

m_full= np.unique(t)

m_full_digitized= np.digitize(m_full, t_bins)

I= {}
for i in range(3):
    I[i]= []
    for j in range(len(m_full_digitized)):
        if m_full_digitized[j] == i:
            I[i].append(j)

A= np.dot(np.dot(S, np.linalg.inv(np.dot(S.T, S))), S.T)

m= np.array([1, 2, 3])

# term 1

# true value
np.dot(np.dot(A, np.dot(S, m)), np.dot(A, np.dot(S, m)))

# check
total= 0.0
for i in range(3):
    total+= np.sum(t_binning == i)*m[i]**2
total

# term 2

# true value
np.dot(np.dot(S, m), np.dot(S, m))

# check
total= 0.0
for i in range(3):
    total+= np.sum(t_binning == i)*m[i]**2
total

# term 3

# true value
np.dot(np.dot(A, np.dot(S, m)), np.dot(S, m))

# check
total= 0.0
for i in range(3):
    total+= np.sum(t_binning == i)**2 * m[i]**2
total






# next check
means= []
for i in range(3):
    means.append(np.sum(t[t_binning == i]))
means= np.array(means)
np.dot(means, m)

np.dot(np.dot(A, t), np.dot(A, np.dot(S, m)))




#########
np.dot(np.dot(A, t), np.dot(S, m))


np.dot(np.dot(A, np.dot(S, m)), t)


np.dot(t, np.dot(S, m))





np.dot(np.dot(A, np.dot(S_full, m_full)), np.dot(S_full, m_full))
np.dot(np.dot(S_full, m_full), np.dot(S_full, m_full))
np.dot(np.dot(A, np.dot(S_full, m_full)), np.dot(A, np.dot(S_full, m_full)))

np.sum(np.multiply(A, np.dot(np.dot(S_full, np.outer(m_full, m_full)), S_full.T)))


total= 0.0
for i in range(3):
    tmp= 0.0
    for j in I[i]:
        tmp+= 1.0/np.sum(t_binning == i)*np.sum(t_binning_full == j)*m_full[j]**2
        #tmp+= m_full[j]**2
    #total+= np.sum(t_binning == i)*m_full[j]**2
    total+= tmp
total



total= 0.0
for i in range(len(A)):
    for j in range(len(A)):
        for k in range(len(m_full)):
            for l in range(len(m_full)):
                total= total + A[i][j]*S_full[j][k]*m_full[k]*S_full[i][l]*m_full[l]
total


total= 0.0
for i in range(len(A)):
    for j in range(len(A)):
        for k, l in itertools.product() 
            total= total + A[i][j]*m_full[k]**2*np.sum(t_binning_full == k)
total

total= 0.0
for i in range(n_bins_full):
    for j in range(n_bins_full):
        if (i in I[0] and j in I[0]):
            total+= 1.0/np.sum(t_binning == 0)*m_full[i]*m_full[j]*np.sum(t_binning_full == i)*np.sum(t_binning_full == j)
        if (i in I[1] and j in I[1]):
            total+= 1.0/np.sum(t_binning == 1)*m_full[i]*m_full[j]*np.sum(t_binning_full == i)*np.sum(t_binning_full == j)
        if (i in I[2] and j in I[2]):
            total+= 1.0/np.sum(t_binning == 2)*m_full[i]*m_full[j]*np.sum(t_binning_full == i)*np.sum(t_binning_full == j)
total






S= np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

A= np.dot(np.dot(S, np.linalg.inv(np.dot(S.T, S))), S.T)


