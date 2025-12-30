#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 15:11:31 2024

@author: cbo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 10:55:08 2024

@author: cbo
"""

import numpy as np
import math
from scipy.stats import poisson
import numpy.random as rnd
import os
import pandas as pd
import math
from sklearn.cluster import KMeans
import numpy as np
from sympy.utilities.iterables import multiset_permutations
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
import networkx as nx
import time
import matplotlib.pyplot as plt
import scipy.sparse as sp
from random import *
from datetime import datetime
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.special import gammaln

def ARI(l_kmeans,partition_generate):
    return(adjusted_rand_score(l_kmeans,partition_generate))

def log_vraisemblance(log_vect_pi,Z,un_X,log_lambda,K,n):
    lambda_ = np.exp(log_lambda)
    #un_X = sp.csr_matrix(mat_adj)
    indices_ones = list(un_X.nonzero())
    numero_cluster = np.argwhere(Z)
    list_partition = np.array([numero_cluster[i,1] for i in range(n) ])
    list_k =list_partition[indices_ones[0]]
    list_l =list_partition[indices_ones[1]]

    if 0 in lambda_[list_k,list_l]:
        
        return(-np.inf)
    else:
        Z_sum = Z.sum(0)

        ll =  (Z_sum @ log_vect_pi).sum()
        #t = Z[indices_ones[0]].reshape(-1, K, 1)* Z[indices_ones[1]].reshape(-1, 1, K)*log_lambda.reshape(1, K, K)
        #t_prim=t.sum(1).sum(1)
        t = log_lambda[list_k,list_l]
        u = t@ un_X[indices_ones[0],indices_ones[1]].T
        #X_log_fact = [math.log(math.factorial(i)) for i in np.array(un_X[indices_ones[0],indices_ones[1]])[0]]
        X_log_fact=gammaln(un_X[indices_ones[0], indices_ones[1]]+1)
        v = u - X_log_fact.sum()
        w = v - (((Z_sum.reshape((-1, 1)) * Z_sum) - Z.T @ Z)* lambda_).sum()
        ll =ll+ w

        return(ll)

def compute_percentile(niveau_theorique,apprentissage,log_pi,partition,log_mat_lambda_1graphe,K,n):
    
    list_vraisemblance_train_graphs = []
    for mongraphe in apprentissage:
        log_v = log_vraisemblance(log_pi,partition,mongraphe,log_mat_lambda_1graphe,K,n)
        list_vraisemblance_train_graphs.append(log_v)
    list_vraisemblance = np.sort(list_vraisemblance_train_graphs)
    print(list_vraisemblance)
    percentile = list_vraisemblance[int(niveau_theorique*len(apprentissage))]
    
    return(percentile)
    
def puissance_star(nb_noeuds,nb_arêtes,test,percentile,log_pi,partition,log_mat_lambda_1graphe,K,n):
    # a essayer ac 5, 10, 15, 20, 25, 30 arêtes
    list_puissance=[]
    dico_graph_attaque = {}
    dico_bad_edges = {}
    for nb_arête in nb_arêtes:
        acceptation= []
        list_attaque = []
        list_bad_edges = []
        for mongraphe in test:
            #istar = randint(0, n-1)
            pte_list_bad_edges = []
            x = np.copy(mongraphe)
            istar = sample([i for i in range(n)], nb_noeuds)
            
            for k in range(nb_noeuds-1):
                pte_list_bad_edges.append((istar[0],istar[k+1]))
                x[istar[0],istar[k+1]]+=nb_arête
            ll = log_vraisemblance(log_pi,partition,x,log_mat_lambda_1graphe,K,n)
            acceptation.append((ll> percentile))
            list_attaque.append(x)
            list_bad_edges.append(pte_list_bad_edges)
        dico_graph_attaque[nb_arête]=list_attaque
        dico_bad_edges[nb_arête]=list_bad_edges
        dico =dict((l, acceptation.count(l)) for l in set(acceptation))
        if False in dico.keys():
            puissance= (dico[False]/len(acceptation))*100
        else:
            puissance=0
        list_puissance.append(puissance)
    print(list_puissance)
    return(list_puissance,dico_graph_attaque,dico_bad_edges)

def puissance_edgenotnull(apprentissage,test, percentile,nb_branche, nb_arêtes,log_pi,partition,log_mat_lambda_1graphe,K,n):
    # a essayer ac 5, 10, 15, 20, 25, 30 arêtes
    X_sum=np.sum(apprentissage,0)
    list_edge_nonnul=[]
    for i in range(X_sum.shape[0]):
       for j in range(X_sum.shape[0]): 
           if X_sum[i,j]!=0:
               list_edge_nonnul.append((i,j))
    list_puissance=[]
    dico_graph_attaque = {}
    dico_bad_edges = {}
    for nb_arête in nb_arêtes:
        acceptation= []
        list_attaque = []
        list_bad_edges = []
        for mongraphe in test:
            #istar = randint(0, n-1)
            pte_list_bad_edges = []
            x = np.copy(mongraphe)
            istar = sample(list_edge_nonnul, nb_branche)
            
            for k in range(nb_branche):
                pte_list_bad_edges.append(istar[k])
                x[istar[k][0],istar[k][1]]+=nb_arête
            ll = log_vraisemblance(log_pi,partition,x,log_mat_lambda_1graphe,K,n)
            acceptation.append((ll> percentile))
            list_attaque.append(x)
            list_bad_edges.append(pte_list_bad_edges)
        dico_graph_attaque[nb_arête]=list_attaque
        dico_bad_edges[nb_arête]=list_bad_edges
        dico =dict((l, acceptation.count(l)) for l in set(acceptation))
        if False in dico.keys():
            puissance= (dico[False]/len(acceptation))*100
        else:
            puissance=0
        list_puissance.append(puissance)
    print(list_puissance)
    return(list_puissance,dico_graph_attaque,dico_bad_edges)
    
def vraisemblance_edges(apprentissage,list_partition,log_pi,log_mat_lambda_1graphe,mat_lambda,n) :   
    
    X = apprentissage
    G=len(apprentissage)
    dico_ij = {}
    
    for i in range(n):
        for j in range(n):
            
            if i!=j:
                list_ll_ij = []
                Z_i = list_partition[i]
                Z_j = list_partition[j]
                lambda_Z_i_Z_j = mat_lambda[Z_i,Z_j]
                log_lambda_Z_i_Z_j = log_mat_lambda_1graphe[Z_i,Z_j]
                log_pi_i = log_pi[Z_i]
                log_pi_j = log_pi[Z_j]
                if lambda_Z_i_Z_j==0:
                    ll=log_pi_i+log_pi_j
                    list_ll_ij = [ll for t in range(G)]
                else:
                    for t in range(G):
                        x = X[t][i,j]
                        if x == 0:
                            
                            ll = log_pi_i+log_pi_j-lambda_Z_i_Z_j
                        if x!=0:
                            ll= log_pi_i+log_pi_j-lambda_Z_i_Z_j + x*log_lambda_Z_i_Z_j-gammaln(x+1)
                            
                        list_ll_ij.append(ll)
                dico_ij[(i,j)]=list_ll_ij    
    return(dico_ij)


def pvalues_edges(dico_graph_attaque, nb_arêtes, apprentissage,partition,log_pi,log_mat_lambda_1graphe,mat_lambda,n,G):
    list_partition =np.argmax(partition,1)
    dico_ij = vraisemblance_edges(apprentissage,list_partition,log_pi,log_mat_lambda_1graphe,mat_lambda,n) 
    dico_pvalue={}
    for nb_arête in nb_arêtes:
        list_list = []
        for g in dico_graph_attaque[nb_arête]:
            attaque = g
            k = 0
            list_pvalue = []
            for i in range(n):
                for j in range(n):
        
                    if i!=j:
        
                        list_ll_ij = dico_ij[(i,j)]
                        k+=1
                        Z_i = list_partition[i]
                        Z_j = list_partition[j]
                        lambda_Z_i_Z_j = mat_lambda[Z_i,Z_j]
                        log_lambda_Z_i_Z_j = log_mat_lambda_1graphe[Z_i,Z_j]
                        log_pi_i = log_pi[Z_i]
                        log_pi_j = log_pi[Z_j]
                        if lambda_Z_i_Z_j==0:
                            x = attaque[i,j]
                            if x ==0:
                                ll=log_pi_i+log_pi_j
                            if x!=0:
                                ll = -np.inf
        
                        else:
                            x = attaque[i,j]
                            if x == 0:
        
                                ll = log_pi_i+log_pi_j-lambda_Z_i_Z_j
                            if x!=0:
                                ll= log_pi_i+log_pi_j-lambda_Z_i_Z_j + x*log_lambda_Z_i_Z_j-gammaln(x+1)
                        pvalue = len([t for t  in list_ll_ij if ll>=t])
                        #if np.min(list_ll_ij)==np.max(list_ll_ij)==ll:
                        #    pvalue=G
                        pvalue = 100*pvalue/G
                        list_pvalue.append(pvalue)
            list_list.append(list_pvalue)
        dico_pvalue[nb_arête]=list_list

    return(dico_pvalue)


def curve_roc_edge_allgraphs(dico_pvalue, dico_bad_edges,nb_arête,n):
    big_list_pvalue = []
    big_list_res = []
    for num_graphe in range(len(dico_pvalue[nb_arête])):
        
        pvalue_0 = dico_pvalue[nb_arête][num_graphe]
        badedges0 = dico_bad_edges[nb_arête][num_graphe]
        pvalue_0 = [i/100 for i in pvalue_0]
        list_res = []
        for i in range(n):
            for j in range(n):
                if i!=j:
                    if (i,j) not in badedges0:
                        list_res.append(1)
                    else:
                        list_res.append(0)
        big_list_res +=list_res
        big_list_pvalue+=pvalue_0
    
    fpr,tpr, thresholds = metrics.roc_curve(big_list_res,big_list_pvalue,pos_label=1)

    auc=metrics.roc_auc_score(big_list_res,big_list_pvalue)
    return(auc,thresholds,fpr,tpr)


def curve_roc_edge(dico_pvalue, dico_bad_edges,nb_arête,num_graphe,n):
    
    pvalue_0 = dico_pvalue[nb_arête][num_graphe]
    badedges0 = dico_bad_edges[nb_arête][num_graphe]
    pvalue0 = [i/100 for i in pvalue_0]
    list_res = []
    for i in range(n):
        for j in range(n):
            if i!=j:
                if (i,j) not in badedges0:
                    list_res.append(1)
                else:
                    list_res.append(0)
    fpr,tpr, thresholds = metrics.roc_curve(list_res,pvalue0,pos_label=1)

    auc=metrics.roc_auc_score(list_res,pvalue0)
    return(auc,thresholds,fpr,tpr)




def log_poisson(x,lambda_):
    log_lambda = np.log(lambda_)
    ll = x*log_lambda
    ll = ll -gammaln(x+1)
    ll = ll-lambda_
    return(ll)


    
def ll_degree_apprentissage(apprentissage,log_tau,tau,mat_lambda,K,n) :

    nb_element_cluster = np.sum(tau,0)
    list_partition = np.argmax(log_tau,axis=1)
    list_lambda_degre = []
    for i in range(n):
        partition_i = list_partition[i]
        lambda_degre = 0
        for l in range(K):
            if l == partition_i:
                lambda_degre+= (nb_element_cluster[l]-1)*mat_lambda[partition_i,l]
            else:
                lambda_degre+= nb_element_cluster[l]*mat_lambda[partition_i,l]
        list_lambda_degre.append(lambda_degre) 
        
    proba_observee_degre_train = np.zeros((n,len(apprentissage)))
    ind_app = 0
    for g in apprentissage:
        degre_observe = np.sum(g,1)
        for i in range(n):
            lambda_degre = list_lambda_degre[i]
            degre_i = degre_observe[i]
    
            ll = log_poisson(degre_i,lambda_degre)
            proba_observee_degre_train[i,ind_app]=ll
        ind_app+=1

    return(list_lambda_degre,proba_observee_degre_train)

def ll_degree_test(apprentissage,log_tau,tau,mat_lambda,dico_graph_attaque,nb_aretes,K,n):
    G = len(apprentissage)
    list_lambda_degre,proba_observee_degre_train = ll_degree_apprentissage(apprentissage,log_tau,tau,mat_lambda,K,n) 
    dico_pval={}
    for nb_arete in nb_aretes:
        list_pval_multigraphe = []
        for g in dico_graph_attaque[nb_arete]:
            degre_observe = np.sum(g,1)
            list_pval = []
            for i in range(n):
                lambda_degre = list_lambda_degre[i]
                x = degre_observe[i]
                ll = log_poisson(x,lambda_degre)
                #mat_reject[i,ind_val]=(ll>=(percentile))
                pvalue = len([val for val in proba_observee_degre_train[i,:] if val <= ll])/G
                list_pval.append(pvalue)
            list_pval_multigraphe.append(list_pval)
        dico_pval[nb_arete]=list_pval_multigraphe
    return(dico_pval)

def pourcentage_réussite(niveau_theorique, dico_bad_edges,nb_aretes,dico_pval):
    list_puissance_node = []
    for nb_arete in nb_aretes:
        malist=dico_pval[nb_arete]
        bad_node = [dico_bad_edges[nb_arete][i][0][0] for i in range(len(dico_bad_edges[nb_arete]))]
        acceptation_node=[]
        k=0
        
        for i in bad_node:
            if malist[k][i]<niveau_theorique:
                acceptation_node.append(False)
            else:
                acceptation_node.append(True)
            k+=1
        dico =dict((l, acceptation_node.count(l)) for l in set(acceptation_node))
        if False in dico.keys():
            puissance= (dico[False]/len(acceptation_node))*100
        else:
            puissance=0
        list_puissance_node.append(puissance)
    return(list_puissance_node)
    







            