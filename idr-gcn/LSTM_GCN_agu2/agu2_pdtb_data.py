# -*- coding: utf-8 -*-
# @Time    : 2018/9/1 9:46
# @Author  : YXZhang

import spacy
import nltk
import os
import time
import pickle
import numpy as np
import re
from collections import defaultdict
import random
# import gensim
from sklearn.model_selection import train_test_split
import os

nlp = spacy.load('en_core_web_sm')
path = r''

# 修改首字母大写
def normalize_upper_lower(token):
    # 首字母大写
    if token[0].isupper():
        if token.__len__()>=2:
            # 多字母首位大写
            if token[1].islower():
                return token.lower()
            else:
                return token
        # 单字母大写
        else:
            return token.lower()
    else:
        return token

# 分离标点，第一个单词小写等(没有单复数还原)
def tokenize():
    path = r'LSTM_GCN_agu2'
    implicit_samples = pickle.load(open(os.path.join(path, 'new_implicit.pickle'), 'rb'))
    implicit_samples_norm = []
    for sample in implicit_samples:
        arg1 = nlp(sample[1])
        arg2 = nlp(sample[2])

        arg1_tokens = [token.text for token in arg1]
        # 第一位单词小写
        first_token = normalize_upper_lower(arg1_tokens[0])
        arg1_tokens.remove(arg1_tokens[0])
        arg1_tokens.insert(0, first_token)

        arg2_tokens = [token.text for token in arg2]
        first_token = normalize_upper_lower(arg2_tokens[0])
        arg2_tokens.remove(arg2_tokens[0])
        arg2_tokens.insert(0, first_token)

        implicit_samples_norm.append([sample[0], arg1_tokens, arg2_tokens, sample[3]])

    pickle.dump(implicit_samples_norm, open(os.path.join(path, 'Implicit_token.pickle'), 'wb'))

# dictionary frequence
def create_dict_fre():
    implicit_samples = pickle.load(open(os.path.join(path, 'Implicit_token.pickle'), 'rb'))
    row_text = []
    for sample in implicit_samples:
        row_text += sample[1]
        row_text += sample[2]
    fdist = nltk.FreqDist(row_text)
    pickle.dump(fdist, open(os.path.join(path, 'implicit_token_fre.pickle'), 'wb'))

    print(fdist.most_common(10))

# def test1():
#     import gensim
#     fre1 = pickle.load(open(os.path.join(path, 'pdtb/Implicit_token_fre.pickle'), 'rb'))
#     pattens = [re.compile('\d+.\d+'), re.compile('\d+,\d+')]
#     # fre1_new = {k.lower():v for (k,v) in fre1.items() if re.search(pattens[0], k)==None and re.search(pattens[1], k)==None}
#
#     fre1_new = defaultdict(int)
#     for k,v in fre1.items():
#         if re.search(pattens[0], k) == None and re.search(pattens[1], k) == None:
#             fre1_new[k.lower()] += v
#
#     fre2 = pickle.load(open(os.path.join(path, 'text-corpus/tmp/giga_voc_fre.pickle'), 'rb'))
#     fre2_new = defaultdict(int)
#     for k, v in fre2.items():
#         fre2_new[k.lower()] += v
#
#     model = gensim.models.KeyedVectors.load_word2vec_format('~/wcheng/data/embedding/GoogleNews-vectors-negative300.bin', binary=True)
#     voc3 = model.wv.vocab.keys()
#     voc1 = [voc for (voc,fre) in fre1.items()]
#     voc2 = [voc for (voc, fre) in fre2.items()]
#
#     print(voc1.__len__())
#     print(voc2.__len__())
#     print(voc3.__len__())
#     print((set(voc1) - set(voc2)).__len__())
#     print((set(voc1) - set(voc3)).__len__())
#     print()

class pddata():
    def __init__(self, relation=None):
        if os.path.exists(os.path.join(path, 'embedding.pickle')) and os.path.exists(os.path.join(path, 'voc2id.pickle')):
            self.embedding = pickle.load(open(os.path.join(path, 'embedding.pickle'), 'rb'))
            self.voc2id = pickle.load(open(os.path.join(path, 'voc2id.pickle'), 'rb'))

        # 生成所需的embedding和词表
        # else:
        #     voc_fre = pickle.load(open(os.path.join(path, 'Implicit_token_fre.pickle'), 'rb'))
        #     model = gensim.models.KeyedVectors.load_word2vec_format(
        #         '/home2/mxguo/wcheng/data/embedding/GoogleNews-vectors-negative300.bin', binary=True)
        #     self.embedding = []
        #     id2voc = list(set(voc_fre.keys()) & set(model.wv.index2word))
        #     for token in id2voc:
        #         self.embedding.append(model[token])
        #     # 添加pad，0
        #     self.embedding.insert(0, [0.0] * 300)
        #     self.embedding = np.array(self.embedding)
        #     id2voc.insert(0, 'pad')
        #     self.voc2id = {voc: index for index, voc in enumerate(id2voc)}
        #     pickle.dump(self.embedding, open(r'/home2/mxguo/wcheng/data/pdtb/embedding.pickle', 'wb'))
        #     pickle.dump(self.voc2id, open(r'/home2/mxguo/wcheng/data/pdtb/voc2id.pickle', 'wb'))
        self.rel = relation
        # self.samples = None
        self.train = None
        self.test = None
        self.class_layer1 = ['Expansion', 'Contingency', 'Comparison', 'Temporal']

    # 加载语料，划分训练集、验证集、测试集, ids化
    def load(self):
        self.relations = ['Expansion', 'Contingency', 'Comparison', 'Temporal']
        implicit_samples = pickle.load(open(os.path.join(path, 'Implicit_token.pickle'), 'rb'))
        implicit_samples_new = []
        # rel_line => rel_list
        for sample in implicit_samples:
            tmp_rel = []
            for rel in self.relations:
                if rel in sample[0]:
                    tmp_rel.append(rel)
            implicit_samples_new.append([tmp_rel, sample[1], sample[2], sample[3]])

        # Lin: sections 2 - 21, 22, and 23 are used as training, dev, and test sets, respectively.
        # Layer 2 type The most frequent 11 types of relations are selected in the task
        #
        # Pilter: sections 2-20, 0-1, and 21-22 are used as training, dev, and test sets, respectively
        # Layer 1 class
        train = [[multi_rel, arg1, arg2] for multi_rel, arg1, arg2, file in implicit_samples_new if int(file[4:6]) >= 2 and int(file[4:6]) <= 20]
        dev = [[multi_rel, arg1, arg2] for multi_rel, arg1, arg2, file in implicit_samples_new if int(file[4:6]) >= 0 and int(file[4:6]) <= 1]
        test = [[multi_rel, arg1, arg2] for multi_rel, arg1, arg2, file in implicit_samples_new if int(file[4:6]) >= 21 and int(file[4:6]) <= 22]

        if not os.path.exists(os.path.join(path, 'train_dev_test.token')):
            pickle.dump([train,dev,test], open(os.path.join(path, 'train_dev_test.token'), 'wb'))

        # rel_str, arg1_id, arg2_id
        train_new = []
        for sample in train:
            # 一个样本最多只有两个label
            if len(sample[0])==2:
                train_new.append([sample[0][1], sample[1], sample[2]])
            train_new.append([sample[0][0], sample[1], sample[2]])

        # rel_list, arg1_id, arg2_id
        dev_new = dev

        # rel_list, arg1_id, arg2_id
        test_new = test

        self.data = {}
        self.data['train'] = self._trans2id(train_new)
        self.data['dev'] = self._trans2id(dev_new)
        self.data['test'] = self._trans2id(test_new)

        # train:rel, arg1_id, arg2_id  dev,test:rel_list, arg1_id, arg2_id
        pickle.dump(self.data, open(os.path.join(path, 'train_dev_test.ids'), 'wb'))

    # 构建指定关系的二分类模型的正负样本
    def gen_rel_data(self, relation):
        self.data = pickle.load(open(os.path.join(path, 'train_dev_test.ids'), 'rb'))
        train_pos = [[arg1, arg2, 1] for rel, arg1, arg2 in self.data['train'] if rel == relation]
        train_neg = [[arg1, arg2, 0] for rel, arg1, arg2 in self.data['train'] if rel != relation]

        test = []
        for sample in self.data['test']:
            if relation in sample[0]:
                if len(sample[0])==2:
                    # 包含指定rel和其他rel类型
                    test.append([sample[1], sample[2], [0, 1]])
                else:
                    # 只包含指定rel
                    test.append([sample[1], sample[2], [1]])
            else:
                # 不包含指定rel
                test.append([sample[1], sample[2], [0]])
        self.tmp_data = {}
        self.tmp_data['train'] = [train_pos, train_neg]
        self.tmp_data['test'] = test

    # 构建4分类模型样本
    def gen_whole_data(self):
        self.data = pickle.load(open(os.path.join(path, 'train_dev_test.ids'), 'rb'))
        rel2id = {'Expansion':0, 'Contingency':1, 'Comparison':2, 'Temporal':3}

        train = [[sample[1], sample[2], rel2id[sample[0]]] for sample in self.data['train']]
        exp = [[sample[0], sample[1], 0] for sample in train if sample[2]==0]
        con = [[sample[0], sample[1], 1] for sample in train if sample[2] == 1]
        com = [[sample[0], sample[1], 2] for sample in train if sample[2] == 2]
        tem = [[sample[0], sample[1], 3] for sample in train if sample[2] == 3]

        self.tmp_data = {}
        self.tmp_data['train'] = [exp, con, com, tem]

        test = [[sample[1], sample[2], [rel2id[rel] for rel in sample[0]]] for sample in self.data['test']]
        self.tmp_data['test'] = test

    # 二分类，指定关系，取一个batch
    def next_single_rel(self, batch_size, type):
        if type == 'train':
            selected_samples = random.sample(self.tmp_data['train'][0], batch_size//2) + \
                               random.sample(self.tmp_data['train'][1], batch_size//2)

        elif type == 'test':
            selected_samples = self.tmp_data['test']
        else:
            return None

        arg_len = np.array([[len(arg1), len(arg2)] for arg1,arg2,label in selected_samples])
        arg1_len = arg_len[:, 0]
        arg2_len = arg_len[:, 1]
        arg1_max_len = max(arg1_len)
        arg2_max_len = max(arg2_len)
        tmp = [[self._padding(arg1, arg1_max_len), self._padding(arg2, arg2_max_len), label]
                        for arg1, arg2, label in selected_samples]
        arg1 = [arg1 for arg1, _, _ in tmp]
        arg2 = [arg2 for _, arg2, _ in tmp]
        label = [label for _, _, label in tmp]
        # print('arg1',arg1)
        return arg1, arg2, label, arg1_len, arg2_len

    def next_multi_rel(self, batch_size, data_type='train', is_balance=True):
        # is_balance:是否获取各label均匀的训练样本。Flase，则只随机获取某一单一label的样本
        if data_type=='train':
            if is_balance:
                selected_samples = random.sample(self.tmp_data['train'][0], batch_size/4) + \
                                   random.sample(self.tmp_data['train'][1], batch_size/4) + \
                                   random.sample(self.tmp_data['train'][2], batch_size/4) + \
                                   random.sample(self.tmp_data['train'][3], batch_size/4)

            else:
                rel_index = random.sample([0,1,2,3], 1)[0]
                selected_samples = random.sample(self.tmp_data['train'][rel_index], batch_size)

        elif data_type=='test':
            selected_samples = self.tmp_data['test']
        else:
            return None

        arg_len = np.array([[len(arg1), len(arg2)] for arg1, arg2, label in selected_samples])
        arg1_len = arg_len[:, 0]
        arg2_len = arg_len[:, 1]
        arg1_max_len = max(arg1_len)
        arg2_max_len = max(arg2_len)
        tmp = [[self._padding(arg1, arg1_max_len), self._padding(arg2, arg2_max_len), label]
                        for arg1, arg2, label in selected_samples]
        arg1 = [arg1 for arg1, _, _ in tmp]
        arg2 = [arg2 for _, arg2, _ in tmp]
        label = [label for _, _, label in tmp]

        if type=='train' and not is_balance:
            return arg1, arg2, arg1_len, arg2_len, rel_index
        else:
            return arg1, arg2, arg1_len, arg2_len, label

    # 跳过未登录词
    def _trans2id(self, samples):
        result = []
        for rel, arg1, arg2 in samples:
            arg1_ids = []
            arg2_ids = []
            for token in arg1:
                id = self.voc2id.get(token, -1)
                if id != -1:
                    arg1_ids.append(id)
            for token in arg2:
                id = self.voc2id.get(token, -1)
                if id != -1:
                    arg2_ids.append(id)
            result.append([rel, arg1_ids, arg2_ids])
        return result

    def _padding(self, ids, max_len):
        # 'pad':0
        return (ids + [0]*max_len)[:max_len]
'''
if __name__ == '__main__':
    # data = pickle.load(open(os.path.join(path, 'Implicit.pickle'), 'rb'))

    start_time = time.time()
    # tokenize()
    # test1()
    # create_dict_fre()
    # implicit_samples = pickle.load(open(os.path.join(path, 'Implicit_token.pickle'), 'rb'))
    pd = pdtb_data()
    pd.gen_whole_data()
    # pd.load()
    # pd.gen_rel_data('Expansion')
    # exit()
    pd.gen_whole_data()
    # pd.next_multi_rel(10, 'train')
    # pd.next_multi_rel(10, 'train')
    # pd.next_multi_rel(10, 'test')
    # pd.gen_rel_data('Expansion')
    # pd.next_single_rel(10, 'test')
    # pd.gen_embedding()
    # pd.load()
    for rel in ['Expansion', 'Contingency', 'Comparison', 'Temporal']:
        print(rel)
        pd.gen_rel_data(rel)
        # pd.next_single_rel(10, 'train')
        print()
    exit()
    # pd.gen_whole_data()

    # pd = pdtb_data('Expansion')
    # pd.next(10)
    for rel in ['Expansion', 'Contingency', 'Comparison', 'Temporal']:
        train = pickle.load(open(os.path.join(path, 'pdtb/samples/%s_train.pickle') % rel, 'rb'))
        test = pickle.load(open(os.path.join(path, 'pdtb/samples/%s_test.pickle') % rel, 'rb'))
        print()
    #     print(rel)
    #     print(samples.__len__())

    print('time:%.1f(minute)' %((time.time()-start_time)/60))
'''
