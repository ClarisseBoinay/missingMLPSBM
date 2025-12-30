import numpy as np
import math
import pandas as pd
from sklearn.cluster import KMeans
import scipy.sparse as sp
import numpy.random as rnd
from numpy import linalg as LA
from scipy.special import gammaln

def generate_graph_poisson(vect_pi_generate, mat_lambda_generate, n, K, G):
    belong = rnd.multinomial(1, vect_pi_generate, n)
    all_X = []
    for g in range(G):
        X = np.zeros((n, n))
        for i in range(n):
            index_first = np.where(belong[i] == 1)[0][0]  # k-1
            meslambdas = mat_lambda_generate[index_first]
            for j in range(n):
                if j != i:
                    index_second = np.where(belong[j] == 1)[0][0]
                    monlambda = meslambdas[index_second]
                    X[i, j] = rnd.poisson(monlambda, 1)
        all_X.append(X)
    return(belong, all_X)


