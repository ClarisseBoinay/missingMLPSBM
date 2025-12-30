import numpy as np
import math
import pandas as pd
from sklearn.cluster import KMeans
import scipy.sparse as sp
import numpy.random as rnd
from numpy import linalg as LA
from scipy.special import gammaln


def it_log_tau(tau, mat_lambda, log_lambda_, log_pi_, sum_X,sum_W, prod_X_logW, X_log_fact,sum_log_fact, K, nb_node, G, H=1):
    "calcul du logarithme de tau"
    b = np.array([False])
    nb_it = 0
    log_lambda_bis = np.copy(log_lambda_)
    log_lambda_bis[log_lambda_bis == -np.inf] = 0
    
    while b.all() == False:

        nb_it += 1
        if nb_it == 50:
            print("longue itération")
            break
            

        previous = np.copy(tau)
        log_tau_ = 0
        
        #log_tau_ = log_tau_ + G*(-tau.sum(0) @ mat_lambda + tau @
        #                                   mat_lambda - tau.sum(0) @ mat_lambda.T + tau @ mat_lambda.T)

        
        log_tau_ = log_tau_ + (-sum_W @ tau @ mat_lambda + G* tau @
                                           mat_lambda - sum_W.T @ tau @ mat_lambda.T +G*tau @ mat_lambda.T)

        log_tau_=log_tau_ + prod_X_logW
        #log_tau_ = log_tau_ + G*(-tau.sum(0) @ mat_lambda + tau @
        #                           mat_lambda - tau.sum(0) @ mat_lambda.T + tau @ mat_lambda.T)
        
        
        #0k
        log_tau_ = log_tau_ + ((sum_X @ tau) @ log_lambda_bis.T) + (sum_X.T @ tau) @ log_lambda_bis
        log_tau_=log_tau_-sum_log_fact
        #log_tau_.reshape((nb_node,0,K))

        log_tau_ = log_tau_ + log_pi_.reshape(1, K)
        # print("log_fin_it "+str(log_tau_))

        # normalisation
        monmax = log_tau_.max(axis=1).reshape(nb_node, 1)
        log_tau_ = log_tau_ - monmax - \
                   np.log((np.exp(log_tau_ - monmax)).sum(1)).reshape(nb_node, 1)
        # print("log_tau ap norm "+str(log_tau_))
        #log_tau_=log_tau_.reshape((nb_node,K,0))
        tau = np.exp(log_tau_)
        
        # Trouver les indices du maximum pour chaque ligne
        # Trouver les indices du maximum pour chaque ligne
        max_indices = np.argmax(tau, axis=1)        
        # Créer un tableau de zéros avec la même forme que A
        B = np.zeros_like(tau)
        
        # Placer des 1 aux indices du maximum de chaque ligne
        B[np.arange(tau.shape[0]), max_indices] = 1
        # de même pr previous
        # Trouver les indices du maximum pour chaque ligne
        max_indices = np.argmax(previous, axis=1)
        
        # Créer un tableau de zéros avec la même forme que A
        A = np.zeros_like(previous)
        
        # Placer des 1 aux indices du maximum de chaque ligne
        A[np.arange(previous.shape[0]), max_indices] = 1
        b = (A == B)
    return (log_tau_, tau)