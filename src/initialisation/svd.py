import numpy as np
import math
import pandas as pd
from sklearn.cluster import KMeans
import scipy.sparse as sp
import numpy.random as rnd
from scipy.sparse.linalg import svds
from scipy.special import gammaln


def svd(mat_adj, k):
    U, S, Vh = svds(mat_adj, k=k)
    # print(S)
    U_tilde = U[:, 0:k]
    V_tilde = Vh[0:k, :].T

    S_tilde = np.diag(S[:k])
    S_tilde = np.diag([1 for i in range(k)])
    a = np.dot(U_tilde, S_tilde ** 1 / 2)
    b = np.dot(V_tilde, S_tilde ** 1 / 2)

    res = np.concatenate((a, b), 1)

    kmeans = KMeans(n_clusters=k, random_state=0).fit(res)
    l = kmeans.labels_
    return (l)
