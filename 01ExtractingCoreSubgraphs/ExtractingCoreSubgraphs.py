import scipy.sparse as sp
import scipy.io
import inspect
import tensorflow as tf
from preprocessing import preprocess_graph, sparse_to_tuple
import numpy as np
import random as rd


def extract_data(adj, labels):

    adj = adj.A

    i_1 = 121
    n = 6
    num = np.ones(n).astype(int)
    num_2 = []
    a = 0
    for i_2 in range(labels.size):
        if adj[i_1, i_2] == 1:
            a = a + 1
            if a <= n:
                for i_3 in range(labels.size):
                    if adj[i_2, i_3] == 1 and adj[i_1, i_3] == 0 and i_3 != i_1:
                        adj[i_3, :] = 0
                        adj[:, i_3] = 0
                        adj[i_2, i_3] = 1
                        adj[i_3, i_2] = 1
                        num_2.append(i_3)
                if i_2 != i_1:
                    num[a - 1] = i_2
            else:
                adj[i_2, i_1] = 0
                adj[i_1, i_2] = 0
                for i_4 in range(num.size):  # i_4=0:1
                    if adj[i_2, num[i_4].astype(int)] == 1:
                        adj[i_2, num[i_4].astype(int)] = 0
                        adj[num[i_4].astype(int), i_2] = 0

    Sub_idx = np.zeros(16).astype(int)
    Sub_idx[0] = i_1
    Sub_idx[1:7] = num[0:6]
    num_2 = np.array(num_2)
    m_rand = num_2.shape[0]
    idx_all = rd.sample(range(m_rand), m_rand)
    num_2 = num_2[idx_all]
    Sub_idx[7:16] = num_2[0:9]


    adj = sp.lil_matrix(adj)


    return adj, labels, Sub_idx