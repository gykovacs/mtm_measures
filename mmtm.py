# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 10:56:02 2016

@author: György Kovács
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression

def pwcmtm(t, w, q):
    """
    Piecewise constant approximation of the Matching by Tone Mapping measure.
    
    Let this measure be denoted by D(t,w,q). 1 - D(t,w,q)*var(w) is basically
    the r2 score of regressing w to t by a piecewise constant function.
    
    Args:
        t (np.array): template vector
        w (np.array): window vector
        q (np.array): bin boundaries
    
    Returns:
        float: the PWC MTM dissimilarity of t and w
    """
    
    # determining the slice vector (unlike in the papers, this is not a binary
    # matrix but a vector with slice indices)
    slices= np.digitize(t, q, right=False)

    # computing the optimal solution of the least squares approximation \hat\beta
    hat_beta= np.array([np.mean(w[slices == i]) for i in range(1, max(slices) + 1)])
    
    # reconstructing the (S . \hat\beta) vector (the vector approximating w)
    hat_w= hat_beta[slices-1]
    
    # computing the measure
    return np.sum((w - hat_w)**2)/(np.var(w)*len(w))

def calculateQ(t, q):
    """
    This function prepares the PWL slice transform matrix
    
    Args:
        t (np.array): template vector
        q (np.array): bin boundaries
        
    Returns:
        np.matrix: the PWL slice transform matrix
    """
    slices= np.zeros([len(t), len(q)], dtype=float)
    
    for i in range(len(t)):
        for j in range(len(q)):
            if ( t[i] < q[j] ):
                slices[i,j-1]= (q[j] - t[i])/(q[j] - q[j-1])
                slices[i,j]= 1 - slices[i,j-1]
                break
    
    return slices
    
def pwlmtm(t, w, q):
    """
    Piecewise linear approximation of the Matching by Tone Mapping measure.
    
    Let this measure be denoted by D(t,w,q). 1 - D(t,w,q)*var(w) is basically
    the r2 score of regressing w to t by a piecewise constant function.
    
    Args:
        t (np.array): template vector
        w (np.array): window vector
        q (np.array): bin boundaries
    
    Returns:
        float: the PWC MTM dissimilarity of t and w
    """

    # computing the slice transform matrix
    Q= calculateQ(t, q)
    
    # determining hat_zeta
    hat_zeta= np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(Q), Q)), np.transpose(Q)), w)
    Qhat_zeta= np.dot(Q, hat_zeta)
    
    # calculating the measure
    return np.sum((w - Qhat_zeta)**2)/(np.var(w)*len(t))

def pwcmmtm(t, w, q):
    """
    Piecewise constant approximation of the Matching by Monotonic Tone Mapping measure
    There is no need to compute the slice transform matrix S explicitly,
    \hat{beta} is computed directly.
    
    Let this measure be denoted by D(t,w,q). 1 - D(t,w,q)*var(w) is basically
    the r2 score of regressing w to t by a piecewise constant function.
    
    Args:
        t (np.array): template vector
        w (np.array): window vector
        q (np.array): bin boundaries
        
    Returns:
        float: the PWC MTM dissimilarity of t and w
    """
    
    # determining the slice vector (unlike in the papers, this is not a binary
    # matrix but a vector with slice indices)
    slices= np.digitize(t, q, right=False)
    # computing the optimal solution of the least squares approximation \hat\beta
    hat_beta, n= zip(*((np.mean(w[slices == i]), np.sum(slices == i)) for i in range(1, max(slices) + 1)))
    hat_beta, n= np.array(hat_beta), np.array(n)
    
    # applying PAVA from the scikit learning package
    ir= IsotonicRegression()
    hat_gamma= ir.fit(range(len(hat_beta)), hat_beta, sample_weight=n).predict(range(len(hat_beta)))
    
    # reconstructing the (S . \hat\beta) vector (the vector approximating w)
    hat_w= hat_gamma[slices-1]
    
    # computing the PWC MMTM dissimilarity score
    return np.sum((w - hat_w)**2)/(np.var(w)*len(w))

def pwlmmtm(t, w, q):
    """
    Piecewise linear approximation of the Matching by Monotonic Tone Mapping measure
    There is no need to compute the slice transform matrix S explicitly,
    \hat{beta} is computed directly.
    
    Args:
        t (np.array): template vector
        w (np.array): window vector
        q (np.array): bin boundaries
        
    Returns:
        float: the PWL MTM dissimilarity of t and w
    """
    
    # determining the slice vector (unlike in the papers, this is not a binary
    # matrix but a vector with slice indices)
    slices= np.digitize(t, q, right=False)
    # computing the optimal solution of the least squares approximation \hat\beta
    hat_beta, n= zip(*((np.mean(w[slices == i]), np.sum(slices == i)) for i in range(1, max(slices) + 1)))
    hat_beta, n= np.array(hat_beta), np.array(n)
    
    # applying PAVA from the scikit learning package
    ir= IsotonicRegression()
    hat_gamma= ir.fit(range(len(hat_beta)), hat_beta, sample_weight=n).predict(range(len(hat_beta)))
    
    # computing the penalty term, which is the PWC representation error
    penalty= np.dot(n, (hat_gamma - hat_beta)**2)/(np.var(w)*len(w))
    
    # computing the original PWL MTM dissimilarity score
    # this component of the PWL MMTM measure is responsible
    # for the handling of fine details
    pwlmtm_value= pwlmtm(t, w, q)

    # computing the total dissimilarity score
    return pwlmtm_value + (1.0 - pwlmtm_value)*penalty

