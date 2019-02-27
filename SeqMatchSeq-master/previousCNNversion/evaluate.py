import numpy as np
import torch
from torch.autograd import Variable
import copy
from metrics import MAP, MRR


def predict(model, data_raw, rl_predict=False):
    data_q, data_as, label = data_raw
    # label = Variable(label, requires_grad=False)
    data_q = data_q.cuda()
    for k in range(len(data_as)):
        data_as[k] = data_as[k].cuda()
    data_as_len = torch.cuda.IntTensor(len(data_as))
    for k in range(len(data_as)):
        data_as_len[k] = data_as[k].size()[0]
        if data_as_len[k] < model.window_large:
            as_tmp = torch.cuda.LongTensor(model.window_large).fill_(0)
            as_tmp[0:data_as_len[k]] = copy.deepcopy(data_as[k])
            data_as[k] = as_tmp
            data_as_len[k] = model.window_large
    # get predicted results
    if rl_predict == False:
        _, new_score = model.comp_agg(data_q, data_as)
        map = MAP(label, new_score.data)
        mrr = MRR(label, new_score.data)
    else:
        new_score = model(data_q, data_as)
        map = MAP(label, new_score.data)
        mrr = MRR(label, new_score.data)
    return map, mrr


def predict_dataset(model, train_dataset, dev_dataset, test_dataset, rl_predict=False):
    model.proj_modules.eval()
    model.dropout_modules.eval()
    model.emb_vecs.eval()
    model.conv_module.eval()

    datasets = [dev_dataset, test_dataset]
    results = []
    for dataset in datasets:
        res = [0., 0.]
        dataset_size = len(dataset)
        for j in range(dataset_size):
            prediction = predict(model, dataset[j], rl_predict=rl_predict)
            res[0] = res[0] + prediction[0]
            res[1] = res[1] + prediction[1]

        res[0] = res[0] / dataset_size
        res[1] = res[1] / dataset_size
        results.append(res)
    for j, vals in enumerate(results):
        if j == 0:
            print("Dev: MAP: %s, MRR: %s" % (vals[0], vals[1]))
        else:
            print("Test: MAP: %s, MRR: %s" % (vals[0], vals[1]))
    return results
