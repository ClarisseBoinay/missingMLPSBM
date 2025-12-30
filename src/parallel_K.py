import numpy as np
# import math
# import pandas as pd
# from sklearn.cluster import KMeans
# import scipy.sparse as sp
# import numpy.random as rnd
# from numpy import linalg as LA

from .vem_svd import vem_svd
from .critere import icl


def parallel_K(sp_X, nb_node, G, K_test):

    print("K")
    print(K_test)

    log_tau, log_vect_pi, log_mat_lambda, mon_J = vem_svd(sp_X, K_test, nb_node, G, X_log_fact=None)
    print("monJ")
    print(mon_J)
    monicl = icl(mon_J, K_test, nb_node, G)
    print("monicl")
    print(monicl)

    log_mat_lambda = log_mat_lambda #- np.log(G)
        #mat_lambda = np.exp(log_mat_lambda)
    return(K_test, log_tau, log_vect_pi, log_mat_lambda,mon_J,monicl)
   
