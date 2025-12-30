import numpy as np
from scipy.special import gammaln
import scipy.sparse as sp
from .vem import vem
from .initialisation import svd
from .parametres import calcul_log_lambda


def vem_svd(sp_X,sum_X, sum_W, prod_X_logW, sum_to_j, K, nb_node, G, X_log_fact=None,sum_log_fact=None):
    l = svd(sum_X,K)
    tau = 0.01*np.ones((nb_node, K))#np.ones((nb_node, K))*0.01
    tau[[i for i in range(nb_node)], l]= 1-((K-1)*0.01)
    log_tau = np.log(tau)
    # mat_lambda = np.zeros((K, K))
    # node_i = 0
    # for i in l:
    #     node_j = 0
    #     for j in l:
    #         mat_lambda[i, j] += sp_X[0][node_i, node_j]
    #         node_j += 1
    #     node_i += 1
    nb_values = np.sum(tau, 0)
    #diviseur = nb_values.reshape(-1, K, 1)*nb_values
    vect_pi_init = nb_values/nb_node
   #mat_lambda = mat_lambda/diviseur[0]
    log_lambda = calcul_log_lambda(tau, log_tau,sum_X,sum_W, K, nb_node, G)
    malist_J = []
    malist_param = []
    if X_log_fact == None:

        sum_log_fact_ij = np.zeros(nb_node)
        sum_log_fact_ji = np.zeros(nb_node)
        
        for un_X in sp_X:
            coo = un_X.tocoo()
            log_fact_values = gammaln(coo.data + 1)
        
            # Accumulate sum by row (axis=1) and column (after transpose)
            np.add.at(sum_log_fact_ij, coo.row, log_fact_values)
            np.add.at(sum_log_fact_ji, coo.col, log_fact_values)
        
        sum_log_fact = np.array(sum_log_fact_ij + sum_log_fact_ji).reshape(nb_node,1)
        X_log_fact = np.sum(sum_log_fact_ij)


    # if X_log_fact == None:
    #     sum_log_fact_ij =0
    #     sum_log_fact_ji =0
    #     for un_X in sp_X:
    #         indices_ones = list(un_X.nonzero())
    #         unX_log_fact=gammaln(np.array(un_X[indices_ones[0], indices_ones[1]])[0]+1)            
    #         X_fact = sp.csr_matrix((nb_node, nb_node), dtype=np.float64)
    #         X_fact[indices_ones[0], indices_ones[1]] = unX_log_fact
            
    #         sum_log_fact_ij+= X_fact.sum(axis=1)

    #         sum_log_fact_ji+=X_fact.T.sum(1)
    
    #     sum_log_fact =np.array( sum_log_fact_ij+ sum_log_fact_ji)
        
    #     X_log_fact = np.sum(sum_log_fact_ij) tau,log_tau, vect_pi, log_mat_lambda, X,sum_X, W,sum_W, prod_X_logW, sum_to_j, X_log_fact,sum_log_fact, K, nb_node, G
    log_tau, log_vect_pi, log_mat_lambda, J, proba_complete = vem(
        tau,log_tau, vect_pi_init, log_lambda, sum_X, sum_W, prod_X_logW, sum_to_j, X_log_fact,sum_log_fact, K, nb_node, G)
    malist_J.append(J)
    malist_param.append((log_tau, log_vect_pi, log_mat_lambda))
    print(J)
    return(log_tau, log_vect_pi, log_mat_lambda, J, proba_complete)

