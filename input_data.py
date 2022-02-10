import scipy.sparse as sp
import scipy.io
import inspect
import tensorflow as tf
from preprocessing import preprocess_graph, sparse_to_tuple
import numpy as np
import random as rd

flags = tf.app.flags
FLAGS = flags.FLAGS



def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(data_source):
    data = scipy.io.loadmat('data/PubMed/Pubmed_0.mat')
    labels = data["Label"]
    attributes = sp.csr_matrix(data["Attributes"])
    network = sp.lil_matrix(data["Network"])

    return network, attributes, labels

def format_data(data_source):

    adj, features, labels = load_data(data_source)

    # 对邻接矩阵进行修改
    adj = adj.A #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # features = features.A

    i_1 = 121  # 选择要进行操作的目标节点%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    n = 6  # 取几个一跳节点组成核心子图%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    num = np.ones(n).astype(int)  # 把要选取子图的所有一跳节点索引保存在num数组中
    num_2 = []  # 把要选取子图的所有二跳节点索引保存在num_2数组中
    a = 0
    for i_2 in range(labels.size):  # 按列进行遍历一跳节点
        if adj[i_1, i_2] == 1:
            a = a + 1
            if a <= n:
                for i_3 in range(labels.size):  # 遍历二跳节点
                    if adj[i_2, i_3] == 1 and adj[i_1, i_3] == 0 and i_3 != i_1:
                        # adj[i_2, i_3] = 0 #
                        # adj[i_3, i_2] = 0 #
                        adj[i_3, :] = 0  ##
                        adj[:, i_3] = 0  ##
                        adj[i_2, i_3] = 1  ##
                        adj[i_3, i_2] = 1  ##
                        num_2.append(i_3)  #存放二跳节点索引
                if i_2 != i_1:
                    num[a - 1] = i_2
            else:
                adj[i_2, i_1] = 0
                adj[i_1, i_2] = 0
                for i_4 in range(num.size):  # i_4=0:1
                    if adj[i_2, num[i_4].astype(int)] == 1:
                        adj[i_2, num[i_4].astype(int)] = 0
                        adj[num[i_4].astype(int), i_2] = 0

    # 操作--合并一跳二跳节点，组成核心+背景子图索引，提取出RealSubgraph
    Sub_idx = np.zeros(16).astype(int)
    Sub_idx[0] = i_1
    Sub_idx[1:7] = num[0:6]
    num_2 = np.array(num_2)
    m_rand = num_2.shape[0]
    idx_all = rd.sample(range(m_rand), m_rand)
    num_2 = num_2[idx_all]  # 随机打乱顺序
    Sub_idx[7:16] = num_2[0:9]

    RealSubAdj = np.zeros((labels.size, labels.size)).astype(int)
    RealSubAtt = np.zeros((labels.size, 500)).astype(float)
    RealSubTruth = np.zeros((labels.size, 1)).astype(int)

    for i in range(Sub_idx.size):
        RealSubAdj[Sub_idx[i], :] = adj[Sub_idx[i], :]
        RealSubAdj[:, Sub_idx[i]] = adj[:, Sub_idx[i]]
        RealSubAtt[Sub_idx[i], :] = features[Sub_idx[i], :]
        RealSubTruth[Sub_idx[i], :] = labels[Sub_idx[i], :]

    RealSubAdj = np.zeros((16, 16)).astype(int)
    RealSubAtt = np.zeros((16, 500)).astype(float)
    RealSubTruth = np.zeros((16, 1)).astype(int)

    for i in range(Sub_idx.size):
        RealSubAtt[i, :] = features[Sub_idx[i], :]
        RealSubTruth[i, :] = labels[Sub_idx[i], :]

    a = 0
    for i in range(len(Sub_idx)):  # 行
        b = 0
        for j in range(len(Sub_idx)):  # 列
            RealSubAdj[a, b] = adj[Sub_idx[i], Sub_idx[j]]
            b = b + 1
        a = a + 1

    adj = RealSubAdj
    features = RealSubAtt
    labels = RealSubTruth

    adj = sp.lil_matrix(adj) #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # features = sp.csr_matrix(features)

    if FLAGS.features == 0:
        features = sp.identity(features.shape[0])  # featureless

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    adj_label = adj + sp.eye(adj.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    items = [adj, num_features, num_nodes, features_nonzero, adj_norm, adj_label, features, labels]
    feas = {}
    for item in items:
        # item_name = [ k for k,v in locals().iteritems() if v == item][0]]
        item_name = retrieve_name(item)
        feas[item_name] = item

    return feas

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var and "item" not in var_name][0]
