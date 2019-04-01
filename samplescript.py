# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 13:57:43 2016

@author: 212578967
"""

import numpy as np
import textwrap

from .mmtm.py import pwcmtm
from .mmtm.py import pwcmmtm
from .mmtm.py import pwlmtm
from .mmtm.py import pwlmmtm

from .mmtm.py import pwcmtm2, pwlmtm2

# a template vector
t= np.array([1, 2, 1, 3, 4, 3], dtype=float)
print("template vector t: %s" % t)

# the vector of bin bounderies for t
q= np.array([0.5, 2.5, 4.5], dtype=float)
print("vector of bin boundaries q: %s" % q)
print("")

# first, the dissimilarity of t and t is examined
print(textwrap.fill("First test: measuring the dissimilarity of t and t -- a vector from itself", width=60))
print("")
print("PWC MTM: %s" % pwcmtm(t, t, q))
print("PWL MTM: %s" % pwlmtm(t, t, q))
print("PWC MMTM: %s" % pwcmmtm(t, t, q))
print("PWL MMTM: %s" % pwlmmtm(t, t, q))
print("")
print(textwrap.fill("As one can see, the monotonicity constraints do not play a role in this case. The PWC measures are affected by the PWC representation error, since the first slice is approximated by the average of [1, 2, 1], while the second slice is approximated by the average of [3, 4, 3].", width=60))
print("")

# the window vector
w0= np.array([1, 2, 1, 3, 4, 5], dtype=float)
print("window vector w0: %s" % w0)
print("")

#second, the dissimilarity of t and w0
print(textwrap.fill("Second test: measuring the dissimilarity of t and a vector differing in fine details within slices: w0", width=60))
print("")
print("PWC MTM: %s" % pwcmtm(t, w0, q))
print("PWL MTM: %s" % pwlmtm(t, w0, q))
print("PWC MMTM: %s" % pwcmmtm(t, w0, q))
print("PWL MMTM: %s" % pwlmmtm(t, w0, q))
print("")
print(textwrap.fill("Now, the PWC representation error is slightly bigger, since in the second slice the elements [3, 4, 5] having range 2 are approximated by their average. The PWL error comes from the fact that the fine pattern [3, 4, 3] within the second slice of t cannot be transformed to [3, 4, 5] by interpolation. The monotonicity constraints still do not have an effect, since the ordering of the slices in w0 is the same as in t.", width=60))
print("")

# the window vector derived from t by a monotonic tone mapping
w1= np.array([1, 2, 1, 5, 15, 5], dtype=float)
print("window vector w1: %s" % w1)
print("")

#third, the dissimilarity of t and w0
print(textwrap.fill("Third test: measuring the dissimilarity of t and the vector w1 differing monotonically from t", width=60))
print(textwrap.fill("The monotonic mapping is: 1 -> 1, 2 -> 2, 3 -> 5, 4 -> 15", width=60))
print("")
print("PWC MTM: %s" % pwcmtm(t, w1, q))
print("PWL MTM: %s" % pwlmtm(t, w1, q))
print("PWC MMTM: %s" % pwcmmtm(t, w1, q))
print("PWL MMTM: %s" % pwlmmtm(t, w1, q))
print("")
print(textwrap.fill("Now, the PWC representation error is slightly bigger, since in the second slice the elements [5, 15, 5] having range 10 are approximated by their average. The PWL error is lower, as the fine pattern [5, 10, 5] can be approximated by scaling the fine pattern [3, 4, 3] pretty well.", width=60))
print("")

w2= np.array([1, 2, 1, -3, -4, -3], dtype=float)
print("window vector w2: %s" % w2)
print("")

#fourth, the dissimilarity of t and w0
print(textwrap.fill("Fourth test: measuring the dissimilarity of t and the vector w2 differing in a non monotonical mapping", width=60))
print(textwrap.fill("the monotonic mapping is: 1 -> 1, 2 -> 2, 3 -> -3, 4 -> -4", width=60))
print("")
print("PWC MTM: %s" % pwcmtm(t, w2, q))
print("PWL MTM: %s" % pwlmtm(t, w2, q))
print("PWC MMTM: %s" % pwcmmtm(t, w2, q))
print("PWL MMTM: %s" % pwlmmtm(t, w2, q))
print("")
print(textwrap.fill("The PWC and PWL MTM measures give relatively low score. PWC MTM is the lowest, as the structure of the vectors is the same and the mapping can be approximated pretty well by an the x -> -x transform. The PCL MTM score is higher, as the fine patterns differ more: in the first slice they can be derived from t by a tone mapping like x -> x, while in the second slice the optimal tone mapping is something like x -> -x. As hat_zeta represents the knot points of the PWL tone mapping (and the slices are connected by these knot points), the second element of hat_zeta should be 2.5 and -2.5 in the same time. Since this is not possible, PWL MTM will deviate from the exact match. The PWC MMTM measure gives the highest possible dissimilarity score, indicating that considering only the mean values of the slices, the relationship between t and w2 is inversely proportional. Similarly, PWL MMTM gives a high dissimilarity score, however, it takes into account that the mapping is not entirely inversely proportional, since some fine details in the slices can still be reconstructed.", width=60))
