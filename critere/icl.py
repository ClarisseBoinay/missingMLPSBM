import numpy as np
import math
import pandas as pd
from sklearn.cluster import KMeans
import scipy.sparse as sp
import numpy.random as rnd
from numpy import linalg as LA
from scipy.special import gammaln

def icl(log_likelihood, K, nb_node, G):

    icl = log_likelihood
    icl = icl - (K**2)/2*np.log(G*nb_node*(nb_node-1))
    icl += - ((K-1)/2)*np.log(nb_node)

    return(icl)