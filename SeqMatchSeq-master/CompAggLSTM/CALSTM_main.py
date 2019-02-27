# Training settings
import argparse
import os
import torch
import random

import loadFiles as tr
from compAggLSTM import compAggWikiqa
from pretrain_train import super_pretrain,train
from evaluate import predict_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=10, help='number of sequences to train on in parallel')
parser.add_argument('--max_epochs', type=int, default=120, help='number of full passes through the training data')
parser.add_argument('--pretrain_epochs', type=int, default=100, help='number of full passes through the training data')
parser.add_argument('--seed', type=int, default=321, help='torch manual random number generator seed')
parser.add_argument('--reg', type=float, default=0, help='regularize value')
parser.add_argument('--emb_lr', type=float, default=0., help='embedding learning rate')
parser.add_argument('--emb_partial', type=bool, default=True, help='only update the non-pretrained embeddings')
parser.add_argument('--lr', type=float, default=4e-4, help='learning rate decay ratio')
parser.add_argument('--lr_decay', type=float, default=0.95, help='learning rate decay ratio')
parser.add_argument('--dropoutP', type=float, default=0.5, help='dropout ratio')
parser.add_argument('--expIdx', type=int, default=0, help='experiment index')
parser.add_argument('--num_classes', type=int, default=2, help='number of classes')

parser.add_argument('--wvecDim', type=int, default=300, help='embedding dimension')
parser.add_argument('--mem_dim', type=int, default=150, help='state dimension')
# parser.add_argument('--att_dim', type=int, default=150, help='attenion dimension') # The original author doesn't really use this argument
parser.add_argument('--gru_dim', type=int, default=128, help='the dimension of GRU')
parser.add_argument('--gru_layers', type=int, default=1, help='layers of GRU')

parser.add_argument('--window_sizes', type=list, default=[1, 2, 3, 4, 5], help='window sizes')
parser.add_argument('--window_large', type=int, default=5, help='largest window size')

parser.add_argument('--model', type=str, default="compAggWikiqa", help='model')
parser.add_argument('--task', type=str, default="wikiqa", help='task')

parser.add_argument('--comp_type', type=str, default="mul", help='w-by-w type')
parser.add_argument('--visualize', type=bool, default=False, help='visualize')

parser.add_argument('--preEmb', type=str, default="glove", help='Embedding pretrained method')
parser.add_argument('--grad', type=str, default="adamax", help='gradient descent method')

parser.add_argument('--log', type=str, default="nothing", help='log message')
parser.add_argument('--gpu', type=bool, default=True, help='use gpu or not')
parser.add_argument('--pretrained_embed', type=bool, default=True, help='use pretrained embedding or not')
parser.add_argument('--pretrain_type', type=bool, default=False, help='rl_pretrain: true or supervised pretrain:false')
parser.add_argument('--if_pretrain', type=bool, default=True, help='pretrain or not')




if __name__ == '__main__':

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)
    torch.set_num_threads(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    tr.init(opt)
    # loading data
    vocab = tr.loadVocab(opt.task)
    ivocab = tr.loadiVocab(opt.task)
    opt.numWords = len(ivocab)
    print("Vocal size: %s" % opt.numWords)
    print('loading data ..')
    train_dataset = tr.loadData('train', opt.task)
    print("Size of training data: %s" % len(train_dataset))
    dev_dataset = tr.loadData('dev', opt.task)
    print("Size of dev data: %s" % len(dev_dataset))
    test_dataset = tr.loadData('test', opt.task)
    # model
    model = compAggWikiqa(opt).cuda()
    # pretrain
    pretrain_if = opt.if_pretrain
    if pretrain_if == True:
        # mode:['pretrain','train']
        mode = 'pretrain'
        cur_best_MAP = 0.714
        for i in range(opt.pretrain_epochs):
            random.shuffle(train_dataset)
            if opt.pretrain_type == True:
                train(model, train_dataset, opt, i, mode=mode)
            else:
                super_pretrain(model, train_dataset, opt, i)
            results = predict_dataset(model, train_dataset, dev_dataset, test_dataset, rl_predict=opt.pretrain_type)
            for j, vals in enumerate(results):
                if j == 1:
                    test_MAP = vals[0]
            if test_MAP > cur_best_MAP:
                print('saving pretrained parameters')
                print()
                if opt.pretrain_type == True:
                    torch.save(model.state_dict(), 'rl_pre_params.pkl')
                else:
                    torch.save(model.state_dict(), 'super_pre_params.pkl')
                cur_best_MAP = test_MAP
        print('loading pretrained parameters...')
        if opt.pretrain_type == True:
            model.load_state_dict(torch.load('rl_pre_params.pkl'))
        else:
            model.load_state_dict(torch.load('super_pre_params.pkl'))
        predict_dataset(model, train_dataset, dev_dataset, test_dataset, rl_predict=opt.pretrain_type)

    # RL training
    for i in range(opt.max_epochs):
        mode = 'train'
        print('rl training !')
        # random.shuffle(train_dataset)
        train(model, train_dataset, opt, i, mode=mode)
        model.optim_state['learningRate'] = model.optim_state['learningRate'] * opt.lr_decay
        results = predict_dataset(model,train_dataset, dev_dataset, test_dataset, rl_predict=True)
        model.save('../trainedmodel/', opt, results, i)
