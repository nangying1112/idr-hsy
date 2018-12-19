import numpy as np
import pickle
import random
import networkx as nx
import scipy.sparse as sp
# from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import re
import tensorflow as tf

def load_corpus(mode,class_num,pos_class=None):

    # 13046: 7004    3340    1942    760
    print('load_corpus...')
    with open('PDTB_data/train_dev_test.data','rb') as f:
        all_data = pickle.load(f)
    data = []
    if mode == 'train':
        data = all_data[0]
    if mode == 'valid':
        data = all_data[1]
    if mode == 'test':
        data = all_data[2]
    random.shuffle(data)
    # generate labels
    labels = []
    if class_num == 4:
        #'Comparison': [1, 0, 0, 0],'Temporal': [0, 1, 0, 0], 'Contingency': [0, 0, 1, 0],  'Expansion': [0, 0, 0, 1],}
        with open("PDTB_data/4-class/y_{}".format(mode), 'rb') as f:
            labels = pickle.load(f)
    if class_num == 2:
        for one in data:
            if pos_class in one[0]:
                labels.append([1,0])
            else:
                labels.append([0,1])
    # print(len(labels))
    labels = np.array(labels)
    # print(labels[0:100])
    with open("PDTB_data/word2id", 'rb') as f:
        word2id = pickle.load(f)
    with open("PDTB_data/embed",'rb') as f:
        embed = pickle.load(f)
    embed = np.array(embed)
    return data, labels, word2id,embed


# 随机取batch

def get_batch(data,labels,word2id,seq_length,batch_size,iter_num):
    print('loading batch_data...')
    iters_per_epoch = len(data)//batch_size+1
    word_set = []
    word_ids = []
    agu1_pad, agu2_pad, label = [], [], []
    dataset_size = len(data)
    agu1_max_length = 0
    agu2_max_length = 0

    agu1_length, agu2_length = [], []
    batch_ids = []
    point = batch_size * (iter_num%iters_per_epoch)
    if point + batch_size > dataset_size:
        point = dataset_size - batch_size
    for i in range(batch_size):
        id = point
        point = point + 1
        batch_ids.append(id)
        agu1 = data[id][1]
        agu2 = data[id][2]
        agu1_len = len(agu1)
        agu2_len = len(agu2)
        if agu1_max_length < agu1_len:
            agu1_max_length = agu1_len
        if agu2_max_length < agu2_len:
            agu2_max_length = agu2_len
    point = point-batch_size
    for i in range(batch_size):
        pad = 'PAD'
        id = point
        point = point + 1
        batch_ids.append(id)
        agu1 = data[id][1]
        agu2 = data[id][2]
        agu1_len = len(agu1)
        agu2_len = len(agu2)
        agu1_length.append(agu1_len)
        agu2_length.append(agu2_len)
        if agu1_len < agu1_max_length:
            agu1 = agu1 + [pad for i in range(agu1_max_length - agu1_len)]
        if agu2_len < agu2_max_length:
            agu2 = agu2 + [pad for i in range(agu2_max_length- agu2_len)]
        agu1_pad.append(agu1)
        agu2_pad.append(agu2)
        label.append(labels[id])
        # print(id)
        # print(i)
        # print(agu1_pad[i])
        # print(agu2_pad[i])
        # print(PDTB_data[id][0])
        # print(label[i])
    agu1_id, agu2_id = [], []
    for agu in agu1_pad:
        temp = []
        for word in agu:
            temp.append(word2id[word])
        agu1_id.append(temp)
    for agu in agu2_pad:
        temp = []
        for word in agu:
            temp.append(word2id[word])
        agu2_id.append(temp)
    label_np = np.array(label)
    # print(label)
    # print(agu1_id)
    # print(agu2_id)
    print('batch agu1 max length:',agu1_max_length)
    print('batch agu2 max length:',agu2_max_length)

    agu1_np = np.array(agu1_id)
    agu2_np = np.array(agu2_id)
    agu1_length = np.array(agu1_length)
    agu2_length = np.array(agu2_length)

    return agu1_np, agu2_np, label_np, agu1_length, agu2_length



def get_valid_test_batch(data,labels,word2id,seq_length,batch_size,iter_num):
    agu1_pad, agu2_pad, label = [], [], []
    dataset_size = len(data)
    agu1_length, agu2_length = [], []
    batch_ids = []
    point = batch_size * iter_num
    if point+batch_size > dataset_size:
        point = dataset_size-batch_size
    agu1_max_length = 0
    agu2_max_length = 0
    for i in range(batch_size):
        id = point
        point = point + 1
        batch_ids.append(id)
        agu1 = data[id][1]
        agu2 = data[id][2]
        agu1_len = len(agu1)
        agu2_len = len(agu2)
        if agu1_max_length < agu1_len:
            agu1_max_length = agu1_len
        if agu2_max_length < agu2_len:
            agu2_max_length = agu2_len
    point = point-batch_size
    for i in range(batch_size):
        pad = 'PAD'
        id = point
        point = point+1
        batch_ids.append(id)
        agu1 = data[id][1]
        agu2 = data[id][2]
        agu1_len = len(agu1)
        agu2_len = len(agu2)
        agu1_length.append(agu1_len)
        agu2_length.append(agu2_len)
        if agu1_len < agu1_max_length:
            agu1 = agu1 + [pad for i in range(agu1_max_length - agu1_len)]
        if agu2_len < agu2_max_length:
            agu2 = agu2 + [pad for i in range(agu2_max_length - agu2_len)]
        agu1_pad.append(agu1)
        agu2_pad.append(agu2)
        label.append(labels[id])
        # print(id)
        # print(i)
        # print(agu1_pad[i])
        # print(agu2_pad[i])
        # print(PDTB_data[id][0])
        # print(label[i])
    agu1_id, agu2_id = [], []
    for agu in agu1_pad:
        temp = []
        for word in agu:
            temp.append(word2id[word])
        agu1_id.append(temp)
    for agu in agu2_pad:
        temp = []
        for word in agu:
            temp.append(word2id[word])
        agu2_id.append(temp)
    label_np = np.array(label)
    # print(label)
    # print(agu1_id)
    # print(agu2_id)
    agu1_np = np.array(agu1_id)
    agu2_np = np.array(agu2_id)
    agu1_length = np.array(agu1_length)
    agu2_length = np.array(agu2_length)
    print('batch ',iter_num,'agu1 max length:', agu1_max_length)
    print('batch ',iter_num,'agu2 max length:', agu2_max_length)

    return agu1_np, agu2_np, label_np, agu1_length, agu2_length




def construct_feed_dict(agu1,agu2, labels,agu1_seq_length,agu2_seq_length, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['batch_agu1']: agu1})
    feed_dict.update({placeholders['batch_agu2']: agu2})
    feed_dict.update({placeholders['batch_labels']: labels})
    feed_dict.update({placeholders['agu1_seq_length']: agu1_seq_length})
    feed_dict.update({placeholders['agu2_seq_length']: agu2_seq_length})

    return feed_dict


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    # print('before process')
    # print(type(adj))
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    # print(type(sparse_to_tuple(adj_normalized)))
    return sparse_to_tuple(adj_normalized)


def loadWord2Vec(filename):
    """Read Word Vectors"""
    vocab = []
    embd = []
    word_vector_map = {}
    file = open(filename, 'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        if len(row) > 2:
            vocab.append(row[0])
            vector = row[1:]
            length = len(vector)
            for i in range(length):
                vector[i] = float(vector[i])
            embd.append(vector)
            word_vector_map[row[0]] = vector
            # print(word_vector_map)
            # break
    print('Loaded Word Vectors!')
    file.close()
    return word_vector_map


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
# load_corpus('p','train')

'''
# 按类别取batch，16+16+16+16

def get_batch(PDTB_data,ll,word2id,word2vec,seq_length,batch_size):
    #print('loading batch_data...')
    batch = []
    with open('PDTB_data/comp','rb') as f:
        comp = pickle.load(f)
    with open('PDTB_data/exp', 'rb') as f:
        exp = pickle.load(f)
    with open('PDTB_data/temp', 'rb') as f:
        temp = pickle.load(f)
    with open('PDTB_data/cont', 'rb') as f:
        cont = pickle.load(f)
    comp_ids = [i for i in range(len(comp))]
    exp_ids = [i for i in range(len(exp))]
    temp_ids = [i for i in range(len(temp))]
    cont_ids = [i for i in range(len(cont))]
    batch=[]
    for i in range(batch_size // 4):
        batch.append(comp[random.choice(comp_ids)])
        batch.append(exp[random.choice(exp_ids)])
        batch.append(cont[random.choice(cont_ids)])
        batch.append(temp[random.choice(temp_ids)])
    random.shuffle(batch)

    agu1_pad, agu2_pad, label = [],[],[]
    #dataset_size = len(PDTB_data)
    #ids = [i for i in range(dataset_size)]
    #batch_ids = []
    agu1_length, agu2_length = [], []
    pad = 'PAD'
    labels ={'Expansion': [0, 0, 0, 1], 'Contingency': [0, 0, 1, 0],
              'Comparison': [1, 0, 0, 0], 'Temporal': [0, 1, 0, 0]}
    with open("PDTB_data/embed",'rb') as f:
        embed = pickle.load(f)
    embed = np.array(embed)
    for one in batch:
        #id = random.choice(ids)
        # 随机取batch and PADDING
       #batch_ids.append(id)
        agu1 = one[1]
        agu2 = one[2]
        agu1_len = len(agu1)
        agu2_len = len(agu2)
        agu1_length.append(min(agu1_len,seq_length))
        agu2_length.append(min(agu2_len,seq_length))
        # print(agu1_len, agu1)
        if agu1_len < seq_length:
            agu1 = agu1+[pad for i in range(seq_length-agu1_len)]
        else:
            agu1 = agu1[(agu1_len-seq_length):]
        if agu2_len < seq_length:
            agu2 = agu2+[pad for i in range(seq_length-agu2_len)]
        else:
            agu2 = agu2[(agu2_len-seq_length):]
        agu1_pad.append(agu1)
        agu2_pad.append(agu2)
        label.append(labels[one[0]])
        # print(labels[one[0]])
    # agu_id
    agu1_id,agu2_id = [],[]
    for agu in agu1_pad:
        temp = []
        for word in agu:
            temp.append(word2id[word])
            # print(word)
            # print(word2id[word])
            # print('embed:',embed[word2id[word]])
        agu1_id.append(temp)
    for agu in agu2_pad:
        temp = []
        for word in agu:
            temp.append(word2id[word])
        agu2_id.append(temp)
    label_np = np.array(label)
    agu1_np = np.array(agu1_id)
    agu2_np = np.array(agu2_id)
    agu1_length = np.array(agu1_length)
    agu2_length = np.array(agu2_length)
    # for i in range(batch_size):
    #     print(agu1_np[i])
    #     print(agu1_length[i])
    #     print(agu2_np[i])
    #     print(label_np[i])
    # print(agu1_np[0][2])
    # print(embed[agu1_np[0][2]])
    return agu1_np,agu2_np,label_np,agu1_length,agu2_length
'''