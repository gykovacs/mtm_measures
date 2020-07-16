#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 18:33:01 2020

@author: gykovacs
"""

import tqdm
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.stats import ttest_ind, ttest_rel
import matplotlib.pyplot as plt

def generate_t(d):
    """
    Generates a random template
    
    Args:
        d (int): dimensionality of the template
    
    Returns:
        np.array: the template
    """
    typ= np.random.randint(3)
    if typ == 0:
        t= np.random.normal(size=d)
    elif typ == 1:
        t= np.random.rand(d)
    elif typ == 2:
        t= np.hstack([np.random.normal(size=int(d/2)), np.random.normal(loc=2, size=(d - int(d/2)))])
    
    t= (t - np.min(t))/(np.max(t) - np.min(t))
    
    exponent= np.random.randint(1, 4)
    
    if np.random.randint(2) == 0:
        t= t**exponent
    else:
        t= t**(1.0/exponent)
    
    t= (t - np.min(t))/(np.max(t) - np.min(t))
    
    return np.round(t, decimals=3)

def generate_noisy_window(d, sigma= 1):
    """
    Generates a noisy window
    
    Args:
        d (int): the dimensionality of the window
        sigma (float): the standard deviation of the noise
        
    Returns:
        np.array: the noisy window
    """
    return np.random.normal(scale=sigma, size=d)    

def eqw_binning(t, n_bins):
    """
    Carries out equal width binning
    
    Args:
        t (np.array): template to bin
        n_bins (int): number of bins
    
    Returns:
        np.array: the binning vector
    """
    
    t_diff= (np.max(t) - np.min(t))/n_bins
    t_bins= np.hstack([np.array([np.min(t) + t_diff*i for i in range(1, n_bins)]), [np.max(t) + 0.01]])
    t_binning= np.digitize(t, t_bins)
    return t_binning

def eqf_binning(t, n_bins):
    """
    Carries out equal frequency binning
    
    Args:
        t (np.array): template to bin
        n_bins (int): number of bins
    
    Returns:
        np.array: the binning vector
    """
    t_bins= []
    t= sorted(t)
    n_items= int(len(t)/n_bins)

    for i in range(1, n_bins):
        t_bins.append(t[int(i*n_items)])
    t_bins.append(np.max(t) + 0.01)
    t_binning= np.digitize(t, t_bins)
    return t_binning

def generate_S_from_binning(t_binning):
    """
    Generates slice matrix from a binning
    
    Args:
        t_binning (np.array): a binning vector
    
    Returns:
        np.array: the slice matrix
    """
    S= np.zeros(shape=(len(t_binning), len(np.unique(t_binning))))
    for i, t_ in enumerate(t_binning):
        S[i][t_]= 1
    return S

def unique_binning(t):
    """
    Carries out the unique binning for a template
    
    Args:
         t (np.array): a template vector
        
    Returns:
        np.array: the unique binning vector
    """
    diff= np.unique(t)
    diff= diff[1:] - diff[:-1]
    diff = np.min(diff)/2
    return np.digitize(t, np.hstack([np.unique(t) + diff]))

def generate_n_u(t):
    """
    Generates the vector n_u containing the umbers of unique elements
    
    Args:
        t (np.array): a template vector
    
    Returns:
        np.array: the n_u vector
    """
    return np.unique(t, return_counts=True)[1]

def generate_tau(t):
    return np.unique(t)

def generate_S_u(t):
    """
    Generates the S_u matrix for a template
    
    Args:
        t (np.array): the template vector
    
    Returns:
        np.array: the S_u matrix
    """
    t_binning = unique_binning(t)
    return generate_S_from_binning(t_binning)

def generate_distorted_t(t, C, sigma):
    """
    Generates the distorted template
    
    Args:
        t (np.array): the template vector
        C (np.array): the covariance matrix of the distortion
        sigma (float): the standard deviation of the white noise
    
    Returns:
        np.array: the distorted template
    """
    S_u = generate_S_u(t)

    m= np.random.multivariate_normal(mean= np.zeros(len(C)), cov=C)
    noise= generate_noisy_window(len(t), sigma)
    return np.dot(S_u, m) + noise

def generate_A_from_S(S):
    """
    Generates the projection matrix A from S
    
    Args:
        S (np.array): slice matrix
    
    Returns:
        A (np.array): the projection matrix
    """
    return np.dot(np.dot(S, np.linalg.inv(np.dot(S.T, S))), S.T)

def generate_A_from_binning(t_binning):
    """
    Generates the projection matrix from binning
    
    Args:
        t_binning (np.array): the binning matrix
    
    Returns:
        A (np.array): the projection matrix
    """
    return generate_A_from_S(generate_S_from_binning(t_binning))


def kmeans_binning(t, n_bins, n_trials=10):
    """
    Carries out kmeans binning
    
    Args:
        t (np.array): the template
        n_bins (int): the number of bins
        n_trials (int): the number of trials
    
    Returns:
        np.array: the binning vector
    """
    best_clustering = None
    best_score = None
    
    for _ in range(n_trials):
        kmeans= KMeans(n_clusters=n_bins, random_state= np.random.randint(100))
        kmeans.fit(t.reshape(-1, 1))
        score= kmeans.score(t.reshape(-1, 1))
        if best_score is None or score > best_score:
            best_score= score
            best_clustering= kmeans.labels_
    
    clusters= np.unique(best_clustering)
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            if np.mean(t[best_clustering == clusters[i]]) < np.mean(t[best_clustering == clusters[j]]):
                tmp_clustering= best_clustering.copy()
                tmp_clustering[best_clustering == clusters[j]]= clusters[i]
                tmp_clustering[best_clustering == clusters[i]]= clusters[j]
                best_clustering= tmp_clustering
    
    return best_clustering

def generate_C_for_m(t, spherical=False, sigma_m=1.0, eigen_th= 0.01):
    """
    Generates a covariance matrix for distortion
    
    Args:
        t (np.array): the template
        spherical (bool): whether the distortion is spherical
        sigma_m (float): the standard deviation of a spherical distortion
        eigen_th (float): threshold on eigenvalues
    
    Returns:
        np.array: the covariance matrix
    """
    if not spherical:
        n_u= generate_n_u(t)
        tmp= np.random.rand(len(n_u), len(n_u))
        matrix= (tmp + tmp.T)/2.0
        eigv, eigw= np.linalg.eigh(matrix)
        eigv[eigv < eigen_th]= eigen_th
        eigv= eigv*(sigma_m*sigma_m)
        matrix= np.dot(np.dot(eigw.T, np.diag(eigv)), eigw)
        return matrix
    else:
        n_u= generate_n_u(t)
        matrix= np.eye(len(n_u))*(sigma_m*sigma_m)
        return matrix

def block_sum(i, bins, C, n_u):
    """
    Computes the sum of a block for greedy binning
    
    Args:
        i (int): bin index
        bins (np.array): bin boundary vector
        C (np.array): covariance matrix
        n_u (np.array): vector of the number of unique elements
    
    Returns:
        float: the contribution of bin i
    """
    s= 0.0
    for j in range(bins[i], bins[i+1]):
        for k in range(bins[i], bins[i+1]):
            s+= C[j][k]*n_u[j]*n_u[k]
    return s

def row_col_sums(i, b_j, bins, C, n_u):
    """
    Computes the contribution of row and column i for bin_j
    
    Args:
        i (int): index
        b_j (int): bin index
        bins (np.array): bin boundary vector
        C (np.array): covariance matrix
        n_u (np.array): vector of the number of unique elements
    
    Returns:
        float: the contribution
    """
    s= C[i][i]*n_u[i]*n_u[i]
    for j in range(bins[b_j], bins[b_j+1]):
        if i != j:
            s+= (C[i][j] + C[j][i])*n_u[i]*n_u[j]
    return s

def changes(i, step, bins, C, n_u, ns, sums):
    """
    Computes the changes in the objective function and temporary arrays
    
    Args:
        i (int): bin boundary index
        step (int): step being made
        bins (np.array): bin boundary vector
        C (np.array): covariance matrix
        n_u (np.array): vector of the number of unique elements
        ns (np.array): temporary array of numbers
        sums (np.array): temporary array of sums
    
    Returns:
        float, float, float, float, float: the change in the objective function,
                    in sums[i], sums[i-1], ns[i], ns[i-1]
    """
    offset= int((step-1)/2)
    sum_i= sums[i] + (-1)*step*row_col_sums(bins[i]+offset, i, bins, C, n_u)
    sum_im1= sums[i-1] + step*row_col_sums(bins[i]+offset, i-1, bins, C, n_u)
    ns_i = (ns[i] - step*n_u[bins[i]+offset])
    ns_im1 = (ns[i-1] + step*n_u[bins[i]+offset])
    
    change= (sum_i)/ns_i - sums[i]/ns[i] + (sum_im1)/ns_im1 - sums[i-1]/ns[i-1]
    
    return change, sum_i, sum_im1, ns_i, ns_im1

def greedy_binning(t, C, n_bins, maxit= 1000):
    """
    Carries out greedy binning
    
    Args:
        t (np.array): the template vector
        C (np.array): the covariance matrix
        n_bins (int): the number of bins
        maxit (int): the maximum number of iterations
    
    Returns:
        np.array: the binning vector
    """
    b= n_bins
    n_u= generate_n_u(t)
    d= len(n_u)
    cum_n_u= np.hstack([[0], np.cumsum(n_u)])
    tau= np.unique(t)
    tau= np.hstack([tau, [np.max(tau) + 0.1]])
    
    splits= sorted(np.random.randint(1, d, b-1))
    while len(np.unique(splits)) < b-1:
        splits= sorted(np.random.randint(1, d, b-1))    
    bins= np.array([0] + splits + [d])
    
    sums= np.repeat(0.0, n_bins)

    for i in range(n_bins):
        sums[i]= block_sum(i, bins, C, n_u)
    
    ns= np.repeat(0.0, n_bins)
    for i in range(n_bins):
        ns[i]= cum_n_u[bins[i+1]] - cum_n_u[bins[i]]
    
    objective= 0.0
    
    for i in range(n_bins):
        objective+= sums[i]/ns[i]

    cum_n_u= np.hstack([[0], np.cumsum(n_u)])
    
    it= 0
    while True and it < maxit:
        it+= 1
        
        change_obj, change_idx, step_, new_sum_i, new_sum_im1, new_ns_i, new_ns_im1= 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        for i in range(1, n_bins):
            for step in [-1, 0]:
                if ns[i + step] > n_u[bins[i] + step]:
                    change, sum_i, sum_im1, ns_i, ns_im1 = changes(i, step*2 + 1, bins, C, n_u, ns, sums)
                    if change > change_obj:
                        change_obj, change_idx, step_, new_sum_i, new_sum_im1, new_ns_i, new_ns_im1= change, i, step*2 + 1, sum_i, sum_im1, ns_i, ns_im1
        
        if change_obj > 0.0:
            objective= objective + change_obj
            bins[change_idx]+= step_
            sums[change_idx]= new_sum_i
            sums[change_idx-1]= new_sum_im1
            ns[change_idx]= new_ns_i
            ns[change_idx-1]= new_ns_im1
        else:
            break
    
    t_binning= []
    for i in range(len(t)):
        for j in range(len(bins)):
            if t[i] >= tau[bins[j]] and t[i] < tau[bins[j+1]]:
                t_binning.append(j)
    
    return np.array(t_binning)

def pwcmtm(t, w, t_binning):
    """
    Piecewise constant approximation of the Matching by Tone Mapping measure.
    
    Let this measure be denoted by D(t,w,q). 1 - D(t,w,q)*var(w) is basically
    the r2 score of regressing w to t by a piecewise constant function.
    
    Args:
        t (np.array): template vector
        w (np.array): window vector
        t_binning (np.array): the binning vector
    
    Returns:
        float: the PWC MTM dissimilarity of t and w
    """


    # computing the optimal solution of the least squares approximation \hat\beta
    hat_beta= np.array([np.mean(w[t_binning == i]) for i in range(max(t_binning) + 1)])
    
    # reconstructing the (S . \hat\beta) vector (the vector approximating w)
    hat_w= hat_beta[t_binning]
    
    # computing the measure
    return np.sum((w - hat_w)**2)/(np.var(w)*len(w))

def exact_noise_value(d=10, b=3):
    """
    Computes the exact expectation of the dissimilarity of t and the white noise
    
    Args:
        d (int): dimensionality of the window
        b (int): number of bins
    
    Returns:
        float: the expectation
    """
    return (d-b)/(d - 1)

def exact_distorted_value(C, t, A, sigma, n_bins):
    """
    Computes the exact expectation of the dissimilarity of t and the distorted t
    
    Args:
        C (np.array): the covariance matrix
        t (np.array): the template
        A (np.array): the projection matrix
        sigma (float): the standard deviation of the white noise
        n_bins (int): the number of bins
    """
    
    num= 0
    denom= 0
    
    n_u = generate_n_u(t)
    S_u = generate_S_u(t)
    
    num+= np.dot(n_u, np.diag(C)) - np.sum(A*np.dot(S_u, np.dot(C, S_u.T)))
    
    num+= sigma**2*(len(t) - n_bins)
    
    denom+= np.dot(n_u, np.diag(C))/len(t)
    denom-= np.dot(n_u, np.dot(C, n_u))/len(t)**2
    denom+= sigma**2*(1 - 1/len(t))
    denom*= len(t)
    
    return num/denom

def exact_kmeans_value(t, A, sigma, sigma_m, n_bins):
    num= 0
    denom= 0
    
    n_u= generate_n_u(t)
    
    num+= np.dot(np.dot(A, t) - t, np.dot(A, t) - t)
    num+= sigma_m**2*(len(n_u) - n_bins)
    
    num+= sigma**2*(len(t) - n_bins)
    denom+= len(t)*np.var(t)
    denom+= sigma_m**2*(len(n_u) - 1) + sigma**2*(len(t) - 1)
    
    return num/denom

"""
t= generate_t(10)
C= generate_C_for_m(t, spherical=True, sigma_m= 1)
tau= np.unique(t)
t_binning= greedy_binning(t, C + np.outer(tau, tau), 3)
A= generate_A_from_binning(t_binning)
dt= generate_distorted_t(t, C, 1)

print(exact_distorted_value(C + np.outer(tau, tau), t, A, 1, 3))

print(exact_kmeans_value(t, A, 1, 1, 3))


S_u= generate_S_u(t)
n_u= generate_n_u(t)


np.dot(n_u, np.diag(C)) + np.dot(n_u, np.diag(np.outer(tau, tau))) - np.sum(A*np.dot(S_u, np.dot(C, S_u.T))) - np.sum(A*np.dot(S_u, np.dot(np.outer(tau, tau), S_u.T)))

np.sum(A*np.dot(S_u, np.dot(np.outer(tau, tau), S_u.T)))
np.dot(np.dot(A, t), t)

np.dot(n_u, np.diag(C))
np.sum(A*np.dot(S_u, np.dot(C, S_u.T)))

np.dot(S_u, np.dot(C, S_u.T))

sigma_m**2*(len(t) - n_bins)

sigma_m= 1
n_bins= 3
np.dot(np.dot(A, t) - t, np.dot(A, t) - t) + sigma_m**2*(len(t) - n_bins)

sigma= 1.0
sigma**2*(len(t) - n_bins)
"""

def k_bins(t, method):
    """
    Determins the number of bins
    
    Args:
        t (np.array): template
        method (int/str): the binning method
    """
    n= len(np.unique(t))
    
    if isinstance(method, int):
        return min([method, n])
    
    if method == 'square-root':
        return int(np.ceil(np.sqrt(n)))
    if method == 'sturges-formula':
        return int(np.ceil(np.log2(n))) + 1
    if method == 'rice-rule':
        return int(np.ceil(2*n**(1/3)))

########################################
# Spherical distortion, kmeans binning #
########################################

# parameters of the simulation
    
spherical=True

d_lower= 100
d_upper= 1000
sigma_lower= 0.1
sigma_upper= 2.0
sigma_m_lower= 0.1
sigma_m_upper= 2.0

bins= [2, 5, 'square-root', 'sturges-formula', 'rice-rule']
binning_methods= ['eqw', 'eqf', 'kmeans', 'greedy']

n_trials=100

def simulation():
    # initialization
    exact_noise, exact_distortion, exact_kmeans= [], [], []
    d_noise, d_distortion, hits= {}, {}, {}
    ds, bs, b_mods, sigmas, sigma_ms= [], [], [], [], []
    
    for binning in binning_methods:
        d_noise[binning]= []
        d_distortion[binning]= []
        hits[binning]= []
    
    for _ in tqdm.tqdm(list(range(n_trials))):
        d= np.random.randint(d_lower, d_upper)
        sigma= sigma_lower + np.random.rand()*(sigma_upper - sigma_lower)
        sigma_m= sigma_m_lower + np.random.rand()*(sigma_m_upper - sigma_m_lower)
        
        t= generate_t(d)
        
        C= generate_C_for_m(t, spherical, sigma_m)
        A= None
        
        for b in bins:
            b_mod= k_bins(t, b)
            
            binnings= []
            for binning in binning_methods:
                if binning == 'eqw':
                    t_binning = eqw_binning(t, b_mod)
                elif binning == 'eqf':
                    t_binning = eqf_binning(t, b_mod)
                elif binning == 'kmeans':
                    t_binning = kmeans_binning(t, b_mod)
                elif binning == 'greedy':
                    t_binning = greedy_binning(t, C, b_mod)
                    A= generate_A_from_binning(t_binning)
                if len(np.unique(t_binning)) != b_mod:
                    #print(binning, len(np.unique(t_binning)), b_mod)
                    break
                binnings.append(t_binning)
            
            if len(binnings) != len(binning_methods):
                continue
    
            ds.append(d)
            bs.append(b)
            b_mods.append(b_mod)
            sigmas.append(sigma)
            sigma_ms.append(sigma_m)
        
            w_noise= generate_noisy_window(d, sigma)
            w_distorted= generate_distorted_t(t, C, sigma)
            if spherical:
                w_distorted= w_distorted + t
                
            for i, binning_method in enumerate(binning_methods):
                binning= binnings[i]
                mtm_noise= pwcmtm(t, w_noise, binning)
                mtm_distorted= pwcmtm(t, w_distorted, binning)
                
                d_noise[binning_method].append(mtm_noise)
                d_distortion[binning_method].append(mtm_distorted)
                
                if mtm_noise > mtm_distorted:
                    hits[binning_method].append(1)
                else:
                    hits[binning_method].append(0)
                
            exact_noise.append(exact_noise_value(d, b_mod))
            if not spherical:
                exact_distortion.append(exact_distorted_value(C, t, A, sigma, b_mod))
            else:
                tau= generate_tau(t)
                exact_distortion.append(exact_distorted_value(C + np.outer(tau, tau), t, A, sigma, b_mod))
            exact_kmeans.append(exact_kmeans_value(t, A, sigma, sigma_m, b_mod))
    
    results= pd.DataFrame({'d': ds,
                           'b': bs,
                           'b_mods': b_mods,
                           'sigma': sigmas,
                           'sigma_m': sigma_ms,
                           'exact_noise': exact_noise,
                           'exact_distortion': exact_distortion,
                           'exact_kmeans': exact_kmeans,
                           'eqw_noise': d_noise['eqw'],
                           'eqf_noise': d_noise['eqf'],
                           'kmeans_noise': d_noise['kmeans'],
                           'greedy_noise': d_noise['greedy'],
                           'eqw_distortion': d_distortion['eqw'],
                           'eqf_distortion': d_distortion['eqf'],
                           'kmeans_distortion': d_distortion['kmeans'],
                           'greedy_distortion': d_distortion['greedy'],
                           'eqw_hits': hits['eqw'],
                           'eqf_hits': hits['eqf'],
                           'kmeans_hits': hits['kmeans'],
                           'greedy_hits': hits['greedy'],})
    return results

results= simulation()

########################################
# General distortion, greedy binning #
########################################

# parameters of the simulation
    
spherical=False

d_lower= 100
d_upper= 1000
sigma_lower= 0.1
sigma_upper= 2.0
sigma_m_lower= 0.1
sigma_m_upper= 2.0

bins= [2, 5, 'square-root', 'sturges-formula', 'rice-rule']
binning_methods= ['eqw', 'eqf', 'kmeans', 'greedy']

n_trials=100

results_greedy= simulation()

###############################
# matching the approximations #
###############################

grouped= results.groupby(['b']).agg(['mean', 'std'])

grouped= grouped.loc[[2, 5, 'sturges-formula', 'rice-rule', 'square-root']]

from matplotlib.transforms import Affine2D

fig, ax= plt.subplots(figsize=(8, 4))
trans0= Affine2D().translate(-0.2, 0.1) + ax.transData
trans1= Affine2D().translate(-0.1, 0.1) + ax.transData
trans2= Affine2D().translate(-0.0, 0.1) + ax.transData
trans3= Affine2D().translate(0.1, 0.1) + ax.transData
trans4= Affine2D().translate(0.2, 0.1) + ax.transData

plt.errorbar(np.arange(len(grouped)), grouped[('exact_noise', 'mean')], grouped[('exact_noise', 'std')], label='exact MV of temp. and noise', linestyle='-', linewidth=2.0)
plt.errorbar(np.arange(len(grouped)), grouped[('kmeans_noise', 'mean')], grouped[('kmeans_noise', 'std')], label='kmeans based MV of tem. and noise', linewidth=2.0, linestyle='-.')
plt.errorbar(np.arange(len(grouped)), grouped[('exact_distortion', 'mean')], grouped[('exact_distortion', 'std')], label='exact MV of temp. and distorted temp.', linestyle='--', linewidth=2.0)
plt.errorbar(np.arange(len(grouped)), grouped[('exact_kmeans', 'mean')], grouped[('exact_kmeans', 'std')], label='exact kmeans MV of temp. and distorted temp.', linestyle='--', linewidth=2.0)
plt.errorbar(np.arange(len(grouped)), grouped[('kmeans_distortion', 'mean')], grouped[('kmeans_distortion', 'std')], label='kmeans based MV of temp. and distorted temp.', linewidth=2.0, linestyle=':')
plt.errorbar(np.arange(len(grouped)), grouped[('greedy_distortion', 'mean')], grouped[('kmeans_distortion', 'std')], label='kmeans based MV of temp. and distorted temp.', linewidth=2.0, linestyle=':')
plt.legend()
plt.xlabel('number of bins')
plt.ylabel('MV dissimilarity')
plt.title('Comparison of exact approximations and measurements for spherical distortion')
plt.xticks(np.arange(len(grouped)), ['2', '5', 'sturges-formula', 'rice-rule', 'square-root'])


grouped= results_greedy.groupby(['b']).agg(['mean', 'std'])

grouped= grouped.loc[[2, 5, 'sturges-formula', 'rice-rule', 'square-root']]


plt.figure(figsize=(8, 4))
plt.errorbar(np.arange(len(grouped)), grouped[('exact_noise', 'mean')], grouped[('exact_noise', 'std')], label='exact MV of temp. and noise', linestyle='-', linewidth=2.0)
plt.errorbar(np.arange(len(grouped)), grouped[('greedy_noise', 'mean')], grouped[('greedy_noise', 'std')], label='greedy based MV of tem. and noise', linewidth=2.0, linestyle='-.')
plt.errorbar(np.arange(len(grouped)), grouped[('exact_distortion', 'mean')], grouped[('exact_distortion', 'std')], label='exact MV of temp. and distorted temp.', linestyle='--', linewidth=2.0)
plt.errorbar(np.arange(len(grouped)), grouped[('greedy_distortion', 'mean')], grouped[('greedy_distortion', 'std')], label='greedy based MV of temp. and distorted temp.', linewidth=2.0, linestyle=':')
plt.legend()
plt.xlabel('number of bins')
plt.ylabel('MV dissimilarity')
plt.title('Comparison of exact approximations and measurements for general distortions')
plt.xticks(np.arange(len(grouped)), ['2', '5', 'sturges-formula', 'rice-rule', 'square-root'])

########
# hits #
########

grouped= results.groupby(['b']).agg(['mean', 'std'])

grouped= grouped.loc[[2, 5, 'sturges-formula', 'rice-rule', 'square-root']]

plt.figure(figsize=(8, 4))
plt.plot(np.arange(len(grouped)), grouped[('eqw_hits', 'mean')], label='EQW binning', linestyle='-', linewidth=2.0)
plt.plot(np.arange(len(grouped)), grouped[('eqf_hits', 'mean')], label='EQF binning', linewidth=2.0, linestyle='-.')
plt.plot(np.arange(len(grouped)), grouped[('kmeans_hits', 'mean')], label='k-means binning', linestyle='--', linewidth=2.0)
plt.plot(np.arange(len(grouped)), grouped[('greedy_hits', 'mean')], label='greedy binning', linewidth=2.0, linestyle=':')
plt.legend()
plt.xlabel('number of bins')
plt.ylabel('MV dissimilarity')
plt.title('Accuracy of recognition for spherical distortion')
plt.xticks(np.arange(len(grouped)), ['2', '5', 'sturges-formula', 'rice-rule', 'square-root'])


grouped= results_greedy.groupby(['b']).agg(['mean', 'std'])

grouped= grouped.loc[[2, 5, 'sturges-formula', 'rice-rule', 'square-root']]

plt.figure(figsize=(8, 4))
plt.plot(np.arange(len(grouped)), grouped[('eqw_hits', 'mean')], label='EQW binning', linestyle='-', linewidth=2.0)
plt.plot(np.arange(len(grouped)), grouped[('eqf_hits', 'mean')], label='EQF binning', linewidth=2.0, linestyle='-.')
plt.plot(np.arange(len(grouped)), grouped[('kmeans_hits', 'mean')], label='k-means binning', linestyle='--', linewidth=2.0)
plt.plot(np.arange(len(grouped)), grouped[('greedy_hits', 'mean')], label='greedy binning', linewidth=2.0, linestyle=':')
plt.legend()
plt.xlabel('number of bins')
plt.ylabel('MV dissimilarity')
plt.title('Accuracy of recognition for general distortion')
plt.xticks(np.arange(len(grouped)), ['2', '5', 'sturges-formula', 'rice-rule', 'square-root'])

############
# p-values #
############

print(np.mean(results['eqw_hits']), np.mean(results['eqf_hits']), np.mean(results['kmeans_hits']), np.mean(results['greedy_hits']))

p_matrix= np.zeros(shape=(len(binning_methods), len(binning_methods)))

for i, b0 in enumerate(binning_methods):
    for j, b1 in enumerate(binning_methods):
        p_matrix[i][j]= ttest_rel(results[b0 + '_hits'], results[b1 + '_hits'])[1]

print(p_matrix)

print(np.mean(results_greedy['eqw_hits']), np.mean(results_greedy['eqf_hits']), np.mean(results_greedy['kmeans_hits']), np.mean(results_greedy['greedy_hits']))

p_matrix_greedy= np.zeros(shape=(len(binning_methods), len(binning_methods)))

for i, b0 in enumerate(binning_methods):
    for j, b1 in enumerate(binning_methods):
        p_matrix_greedy[i][j]= ttest_rel(results_greedy[b0 + '_hits'], results_greedy[b1 + '_hits'])[1]

print(p_matrix)





