import numpy as np
import math
import pandas as pd
from sklearn.cluster import KMeans
import scipy.sparse as sp
import numpy.random as rnd
from numpy import linalg as LA
from scipy.special import gammaln

def calcul_log_pi(log_tau_, K, nb_node):
    log_pi_ = []

    for k in range(0, K):
        max_log_tau = np.max(log_tau_[:, k])
        monlogpi = -np.log(nb_node) + max_log_tau + \
            np.log(np.sum(np.exp(log_tau_[:, k] - max_log_tau)))
        monlogpi1 = np.copy(monlogpi)
        log_pi_.append(monlogpi1)
    return(log_pi_)
