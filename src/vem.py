import numpy as np
import math
import pandas as pd
from sklearn.cluster import KMeans
import scipy.sparse as sp
import numpy.random as rnd
from numpy import linalg as LA
from scipy.special import gammaln

from .critere import calcul_J
from .parametres import calcul_log_lambda, calcul_log_pi, it_log_tau

def vem(tau,log_tau, vect_pi, log_mat_lambda, sum_X,sum_W, prod_X_logW, sum_to_j, X_log_fact,sum_log_fact, K, nb_node, G):
    list_J = []
    list_param = []

    tau_past = np.copy(tau)
    log_tau_past = log_tau
    log_vect_pi_past = np.log(vect_pi)
    list_proba_complete=[]

    log_lambda_past = np.copy(log_mat_lambda)
    mat_lambda_past = np.exp(log_lambda_past)

    #ind_moinsinf =np.where(log_lambda_past==-np.inf)
    #log_lambda_past[log_lambda_past==-np.inf]=0
 # log_tau, tau, log_vect_pi, lambda_, log_lambda,sum_X,sum_W, sum_to_j, X_log_fact, K, nb_node, G
    monj_past, proba_complete_past = calcul_J(log_tau_past, tau_past, log_vect_pi_past,
                         mat_lambda_past,log_lambda_past, sum_X,sum_W,sum_to_j, X_log_fact, K, nb_node, G)
    print(monj_past)
    list_J.append(monj_past)
    list_proba_complete.append(proba_complete_past)
    list_param.append((log_tau_past, log_vect_pi_past, log_lambda_past))
    u = monj_past

    log_tau_next, tau_next = it_log_tau(
        tau_past, mat_lambda_past, log_lambda_past, log_vect_pi_past,sum_X,sum_W, prod_X_logW, X_log_fact, sum_log_fact, K, nb_node, G)
    log_vect_pi_next = calcul_log_pi(log_tau_next, K, nb_node)
    #log_vect_pi_next = calcul_log_pi(log_tau_next,nb_node)

    log_lambda_next = calcul_log_lambda(tau_next,log_tau_next, sum_X, sum_W, K, nb_node, G)
    mat_lambda_next = np.exp(log_lambda_next)
    #log_lambda_next = calcul_log_lambda(tau_next,X,K,nb_node,G)
    v, proba_complete_next = calcul_J(log_tau_next, tau_next, log_vect_pi_next,
                 mat_lambda_next, log_lambda_next, sum_X, sum_W,sum_to_j , X_log_fact, K, nb_node, G)

    it = 0
    list_J.append(v)
    list_proba_complete.append(proba_complete_next)
    list_param.append((log_tau_next, log_vect_pi_next, log_lambda_next))

    critère = False
    while critère == False:
        print("critère J : "+str(v))

        u = np.copy(v)
        it += 1

        tau_past = np.copy(tau_next)
        log_tau_past = np.copy(log_tau_next)
        log_vect_pi_past = np.copy(log_vect_pi_next)
        log_lambda_past = np.copy(log_lambda_next)
        mat_lambda_past = np.copy(mat_lambda_next)

        log_tau_next, tau_next = it_log_tau(
            tau_past, mat_lambda_past, log_lambda_past, log_vect_pi_past, sum_X, sum_W, prod_X_logW, X_log_fact,sum_log_fact, K, nb_node, G)

        log_vect_pi_next = calcul_log_pi(log_tau_next, K, nb_node)
        #print("log_pi_next")
        log_lambda_next = calcul_log_lambda(tau_next,log_tau_next,sum_X,sum_W, K, nb_node, G)
        mat_lambda_next = np.exp(log_lambda_next)
        #mat_lambda_next[mat_lambda_next==np.inf]=4.02890008e+33
        v,proba_complete_next = calcul_J(log_tau_next, tau_next, log_vect_pi_next,
                     mat_lambda_next, log_lambda_next, sum_X, sum_W, sum_to_j, X_log_fact, K, nb_node, G)
        list_J.append(v)
        list_proba_complete.append(proba_complete_next)
        list_param.append((log_tau_next, log_vect_pi_next, log_lambda_next))

        if list_J[-1] == list_J[-3]:
            critère = True
        if np.abs(v-u) < 0.01:
            critère = True
        if it == 50:
            critère = True
    mon_J = max(list_J)
    mesparam = list_param[list_J.index(mon_J)]
    ma_proba_complete=list_proba_complete[list_J.index(mon_J)]
    #mon_J=list_J[-1]
    #mesparam=list_param[-1]
    return(mesparam[0], mesparam[1], mesparam[2], mon_J,ma_proba_complete)

