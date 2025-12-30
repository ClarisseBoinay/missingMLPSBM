import math
import numpy as np
from scipy.special import gammaln


def log_vraisemblance(log_vect_pi, Z, un_X, log_lambda, K, n):
    lambda_ = np.exp(log_lambda)
    # un_X = sp.csr_matrix(mat_adj)
    indices_ones = list(un_X.nonzero())
    numero_cluster = np.argwhere(Z)
    list_partition = np.array([numero_cluster[i, 1] for i in range(n)])
    list_k = list_partition[indices_ones[0]]
    list_l = list_partition[indices_ones[1]]

    if 0 in lambda_[list_k, list_l]:

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
        w = v - (((Z_sum.reshape((-1, 1)) * Z_sum) - Z.T @ Z) * lambda_).sum()
        ll = ll + w

        return (ll)


def vraisemblance_edges(apprentissage, list_partition, log_pi, log_mat_lambda_1graphe, mat_lambda, n,G):
    X = apprentissage
    dico_ij = {}
    N = len(apprentissage)
    for i in range(n):
        for j in range(n):

            if i != j:
                list_ll_ij = []
                Z_i = list_partition[i]
                Z_j = list_partition[j]
                lambda_Z_i_Z_j = mat_lambda[Z_i, Z_j]
                log_lambda_Z_i_Z_j = log_mat_lambda_1graphe[Z_i, Z_j]
                log_pi_i = log_pi[Z_i]
                log_pi_j = log_pi[Z_j]
                if lambda_Z_i_Z_j == 0:
                    ll = log_pi_i + log_pi_j
                    list_ll_ij = [ll for t in range(G)]
                else:
                    for t in range(N):
                        x = X[t][i, j]
                        if x == 0:
                            ll = log_pi_i + log_pi_j - lambda_Z_i_Z_j
                        if x != 0:
                            ll = log_pi_i + log_pi_j - lambda_Z_i_Z_j + x * log_lambda_Z_i_Z_j - gammaln(x+1)#math.log(math.factorial(x))

                        list_ll_ij.append(ll)
                    #G_missing = G - len(list_ll_ij)
                    #nb_to_add = log_pi_i + log_pi_j - lambda_Z_i_Z_j
                    #to_add = [nb_to_add for m in range(G_missing)]
                    #list_ll_ij += to_add
                dico_ij[(i, j)] = list_ll_ij
    return (dico_ij)

def edge_power_row(row,dico_ij,log_pi,log_mat_lambda,mat_lambda,list_partition,G_all):
    i = row[1]
    j = row[2]
    count = row[3]
    list_ll_ij = dico_ij[(i, j)]
    Z_i = list_partition[i]
    Z_j = list_partition[j]
    lambda_Z_i_Z_j = mat_lambda[Z_i, Z_j]
    log_lambda_Z_i_Z_j = log_mat_lambda[Z_i, Z_j]
    log_pi_i = log_pi[Z_i]
    log_pi_j = log_pi[Z_j]
    if lambda_Z_i_Z_j == 0:
        x = count
        if x == 0:
            ll = log_pi_i + log_pi_j
        if x != 0:
            ll = -np.inf

    else:
        x = count
        if x == 0:
            ll = log_pi_i + log_pi_j - lambda_Z_i_Z_j
        if x != 0:
            ll = log_pi_i + log_pi_j - lambda_Z_i_Z_j + x * log_lambda_Z_i_Z_j - gammaln(x+1)
    #ll_0 = log_pi_i + log_pi_j - lambda_Z_i_Z_j
    pvalue = len([t for t in list_ll_ij if ll > t])

    pvalue = 100 * pvalue / len(list_ll_ij)
    return(pvalue)

def pvalues_edges(df_test, apprentissage, partition, log_pi, log_mat_lambda, mat_lambda,
                  n, G):
    list_partition = np.argmax(partition, 1)
    dico_ij = vraisemblance_edges(apprentissage, list_partition, log_pi, log_mat_lambda, mat_lambda, n,G)
    df_test["pvalue"]=df_test.apply(edge_power_row,axis=1,args=(dico_ij,log_pi,log_mat_lambda,mat_lambda,list_partition,G))
    return(df_test)

################ DEGREE

def log_poisson(x,lambda_):
    if x==0:
        return(-lambda_)
    else:
        log_lambda = np.log(lambda_)
        ll = x*log_lambda
        ll = ll -gammaln(x+1)
        ll = ll-lambda_
        return(ll)


def ll_degree_apprentissage(apprentissage, log_tau, tau, mat_lambda, K, n, G):
    nb_element_cluster = np.sum(tau, 0)
    list_partition = np.argmax(log_tau, axis=1)
    list_lambda_degre = []
    for i in range(n):
        partition_i = list_partition[i]
        lambda_degre = 0
        for l in range(K):
            if l == partition_i:
                lambda_degre += (nb_element_cluster[l] - 1) * mat_lambda[partition_i, l]
            else:
                lambda_degre += nb_element_cluster[l] * mat_lambda[partition_i, l]
        list_lambda_degre.append(lambda_degre)

    proba_observee_degre_train = np.zeros((n, len(apprentissage)))
    #proba_observee_degre_train =[]
#    for i in range(n):
#        proba_observee_degre_train[i,:]=-list_lambda_degre[i]
    ind_app = 0
    for g in apprentissage:
        degre_observe = np.sum(g, 1)
        for i in range(n):
            lambda_degre = list_lambda_degre[i]
            degre_i = degre_observe[i][0,0]

            ll = log_poisson(degre_i, lambda_degre)
            proba_observee_degre_train[i, ind_app] = ll
        ind_app += 1
    

    return (list_lambda_degre, proba_observee_degre_train)


def node_power_row(row, list_lambda_degre, proba_observee_degre_train, G):
    i = row[1]
    x = row[2]
    lambda_degre = list_lambda_degre[i]

    ll = log_poisson(x, lambda_degre)
    # mat_reject[i,ind_val]=(ll>=(percentile))
    pvalue = len([val for val in proba_observee_degre_train[i, :] if val < ll]) / len(proba_observee_degre_train[i, :])
    return (pvalue)


def pvalues_nodes(df_test, apprentissage, log_tau, tau, mat_lambda, K,
                  n,G):
    list_lambda_degre, proba_observee_degre_train = ll_degree_apprentissage(apprentissage, log_tau, tau, mat_lambda, K,
                                                                            n,G)
    df_test["pvalue"] = df_test.apply(node_power_row, axis=1,
                                           args=(list_lambda_degre, proba_observee_degre_train, G))
    return (df_test)