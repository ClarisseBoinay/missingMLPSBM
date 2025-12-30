import numpy as np
import math
import pandas as pd
from sklearn.cluster import KMeans
import scipy.sparse as sp
import numpy.random as rnd
from numpy import linalg as LA
from scipy.special import gammaln

def calcul_log_lambda(tau, log_tau, sum_X,sum_W, K, nb_node, G):
    log_lambda= np.zeros((K,K))
    for k in range(K):
        for l in range(K):
            max_num = log_tau[0,k]+log_tau[1,l]+math.log(sum_X[1,0])
            max_den = log_tau[0,k]+log_tau[1,l]+math.log(sum_W[1,0])
            for i in range(nb_node):
                for j in range(nb_node):
                    if i!=j:
                        max_num=max([max_num,log_tau[i,k]+log_tau[j,l]+math.log(sum_X[i,j])])
                        max_den=max([max_num,log_tau[i,k]+log_tau[j,l]+math.log(sum_W[i,j])])
                    
            sum_num=0
            sum_den=0
            for i in range(nb_node):
                for j in range(nb_node):
                    if i!=j:
                        sum_num += math.exp(log_tau[i,k]+log_tau[j,l]+math.log(sum_X[i,j])-max_num)
                        sum_den += math.exp(log_tau[i,k]+log_tau[j,l]+math.log(sum_W[i,j])-max_den)
            num=math.log(sum_num)+max_num
            den=math.log(sum_den)+max_den
            log_lambda[k,l]= num-den
    return(log_lambda)
                        
                    
                    

# def calcul_log_lambda(tau, log_tau, sum_X,sum_W, K, nb_node, G):
    
#     log_lambda = np.zeros((K, K))

#     for k in range(K):
#         for l in range(K):
#             # if k == l:
#             #     if len([i for i in np.round(tau[:, k]) if i == 1]) > 1 and len(
#             #             [i for i in np.round(tau[:, l]) if i == 1]) > 1:
#             #         one_node = False
#             #     else:
#             #         one_node = True
#             # if k != l:
#             #     if len([i for i in np.round(tau[:, k]) if i == 1]) == 0 or len(
#             #             [i for i in np.round(tau[:, l]) if i == 1]) == 0:
#             #         one_node = True
#             #     else:
#             #         one_node = False
#             one_node=False
#             if one_node == False:
                
#                 indices_ones = list(sum_X.nonzero())
#                 monmax = log_tau[indices_ones[0][0], k] + log_tau[indices_ones[1]
#                 [0], l] + np.log(sum_X[indices_ones[0][0], indices_ones[1][0]])
#                 # for un_X in sp_X:
#                 #     # un_X = sp.csr_matrix(un_X)
#                 #     indices_ones = list(un_X.nonzero())
#                 #     monmax1 = np.max(log_tau[indices_ones[0], k] + log_tau[indices_ones[1],
#                 #     l] + np.log(un_X[indices_ones[0], indices_ones[1]]))
#                 #     monmax = max(monmax1, monmax)
                
#                 monmax1 = np.max(log_tau[indices_ones[0], k] + log_tau[indices_ones[1],
#                                                                        l] + np.log(sum_X[indices_ones[0], indices_ones[1]]))
#                 monmax = max(monmax1, monmax)
#                 masom = 0
#                 # for un_X in sp_X:
#                 #     # un_X = sp.csr_matrix(un_X)
#                 #     indices_ones = list(un_X.nonzero())
#                 #     masom += (np.exp((log_tau[indices_ones[0], [k]] + log_tau[indices_ones[1], [
#                 #         l]] + np.log(un_X[indices_ones[0], indices_ones[1]])) - monmax)).sum()
#                 masom += (np.exp((log_tau[indices_ones[0], [k]] + log_tau[indices_ones[1], [
#                         l]] + np.log(sum_X[indices_ones[0], indices_ones[1]])) - monmax)).sum()
#                 masom = np.log(masom) + monmax - np.log(G)

#                 if k == l:
                    
#                     max_log_tau = log_tau[0, k] + log_tau[1, k] + np.log(sum_W[0,1])

#                     for i in range(0, nb_node):
#                         for j in range(0, nb_node):
#                             if i != j:
#                                 max_log_tau = np.max(
#                                     [max_log_tau, log_tau[i, k] + log_tau[j, k] +np.log(sum_W[i,j])])
                

#                     #s = np.argsort(log_tau[:, k])
#                     #max_log_tau = np.sum(log_tau[:, k][s[-2:]]) 

#                 if k != l:
#                     # ne fonctionne pas
#                     # s_1 = np.max(log_tau[:,k])
#                     # s_2 = np.max(log_tau[:,l])
#                     # max_log_tau_1 =s_1+s_2

#                     # max_log_tau=np.sum(log_tau[:,k][s[-2:]])

#                     max_log_tau = log_tau[0, k] + log_tau[1, l] + np.log(sum_W[0,1])

#                     for i in range(0, nb_node):
#                         for j in range(0, nb_node):
#                             if i != j:
#                                 max_log_tau = np.max(
#                                     [max_log_tau, log_tau[i, k] + log_tau[j, l] +np.log(sum_W[i,j])])
                
#                     # if max_log_tau_1==max_log_tau:
#                     #    print("ok")

#                     # max_log_tau= max_log_tau_1
#                     # else:
#                     #    print("notok")
#                 # res2 = np.copy(max_log_tau)
#                 #print("max_log_tau "+str(max_log_tau))
#                 masom2 = 0
#                 for i in range(0, nb_node):
#                     for j in range(0, nb_node):
#                         if j != i:
#                             #print("log tau[ik} "+str(log_tau[i, k]))
#                             #print("log tau jl "+str(log_tau[j, l]))
#                             masom2 += math.exp(log_tau[i, k] +
#                                                log_tau[j, l] +np.log(sum_W[i,j])- max_log_tau)
#                 masom += -(math.log(masom2) + max_log_tau)

#                 log_lambda[k, l] = masom
#             # elif one_node == True:
#             #     log_lambda[k, l] = -np.inf

#     return (log_lambda)

