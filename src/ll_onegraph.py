import numpy as np
from scipy.special import gammaln


def ll_onegraph(log_vect_pi, Z, un_X, log_lambda, mat_lambda, K, n):
    #lambda_ = np.exp(log_lambda)
    # un_X = sp.csr_matrix(mat_adj)
    indices_ones = list(un_X.nonzero())
    numero_cluster = np.argwhere(Z)
    list_partition = np.array([numero_cluster[i, 1] for i in range(n)])
    list_k = list_partition[indices_ones[0]]
    list_l = list_partition[indices_ones[1]]
    
    if len(list_k)==0: 
        Z_sum = Z.sum(0)
        ll = (Z_sum @ log_vect_pi).sum() - (((Z_sum.reshape((-1, 1)) * Z_sum) - Z.T @ Z) * mat_lambda).sum()
        return(ll)
    
    if 0 in mat_lambda[list_k, list_l]:

        return (-np.inf)
    else:
        Z_sum = Z.sum(0)

        ll = (Z_sum @ log_vect_pi).sum()
        # t = Z[indices_ones[0]].reshape(-1, K, 1)* Z[indices_ones[1]].reshape(-1, 1, K)*log_lambda.reshape(1, K, K)
        # t_prim=t.sum(1).sum(1)
        t = log_lambda[list_k, list_l]
        u = t @ un_X[indices_ones[0], indices_ones[1]].T
        # X_log_fact = [math.log(math.factorial(i)) for i in np.array(un_X[indices_ones[0],indices_ones[1]])[0]]
        X_log_fact = gammaln(np.array(un_X[indices_ones[0], indices_ones[1]])[0] + 1)
        v = u - X_log_fact.sum()
        w = v - (((Z_sum.reshape((-1, 1)) * Z_sum) - Z.T @ Z) * mat_lambda).sum()
        ll = ll + w

        return (ll)