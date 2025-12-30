import numpy as np


def calcul_J(log_tau, tau, log_vect_pi, lambda_, log_lambda,sum_X,sum_W, sum_to_j, X_log_fact, K, nb_node, G):
    tau_sum = tau.sum(0)
    log_lambda_bis = np.copy(log_lambda)
    log_lambda_bis[log_lambda_bis==-np.inf]=0
    log_tau_bis = np.copy(log_tau)
    log_tau_bis[log_tau_bis == -np.inf]=0
    ll=0
    #s= G*(((tau_sum.reshape((-1, 1)) * tau_sum) -
    #             tau.T @ tau) * lambda_).sum()
    s = np.einsum('ik,jl,ij,kl ->', tau, tau, sum_W.toarray(), lambda_)

    indices_ones = list(sum_X.nonzero())
    t = tau[indices_ones[0]].reshape(-1, K, 1) * tau[indices_ones[1]
                                                         ].reshape(-1, 1, K)*log_lambda_bis.reshape(1, K, K)
    
    #        t = tau[indices_ones[0]].reshape(-1, K, 1) * tau[indices_ones[1]
                                                       #      ].reshape(-1, 1, K)*log_lambda_bis.reshape(1, K, K)
    u = t.sum(1).sum(1) @ sum_X[indices_ones[0], indices_ones[1]].T
    ll+=u -s + sum_to_j

    ll=ll - X_log_fact
    ll+=(tau_sum @ log_vect_pi).sum()
    proba_complete=np.copy(ll)
    
    ll += (-np.sum(tau * log_tau_bis)).sum()

    return(ll,proba_complete)
