# Training settings
import argparse
import os
import torch
import random

import util.loadFiles as tr
from wikiqa.compAggWikiqa import compAggWikiqa, train,pretrain,rl_pretrain

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=10, help='number of sequences to train on in parallel')
parser.add_argument('--max_epochs', type=int, default=120, help='number of full passes through the training data')
parser.add_argument('--pretrain_epochs', type=int, default=50, help='number of full passes through the training data')
parser.add_argument('--seed', type=int, default=321, help='torch manual random number generator seed')
parser.add_argument('--reg', type=float, default=0, help='regularize value')
parser.add_argument('--learning_rate', type=float, default=0.004, help='learning rate')
parser.add_argument('--emb_lr', type=float, default=0., help='embedding learning rate')
parser.add_argument('--emb_partial', type=bool, default=True, help='only update the non-pretrained embeddings')
parser.add_argument('--lr', type=float, default=4e-4, help='learning rate decay ratio')
parser.add_argument('--lr_decay', type=float, default=0.99, help='learning rate decay ratio')
parser.add_argument('--dropoutP', type=float, default=0.3, help='dropout ratio')
parser.add_argument('--expIdx', type=int, default=0, help='experiment index')
parser.add_argument('--num_classes', type=int, default=2, help='number of classes')

parser.add_argument('--wvecDim', type=int, default=300, help='embedding dimension')
parser.add_argument('--mem_dim', type=int, default=150, help='state dimension')
# parser.add_argument('--att_dim', type=int, default=150, help='attenion dimension') # The original author doesn't really use this argument
parser.add_argument('--cov_dim', type=int, default=150, help='conv dimension')
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
parser.add_argument('--rl_pretrain', type=bool, default=True, help='rl_pretrain or supervised pretrain')



opt = parser.parse_args()
torch.manual_seed(opt.seed)
torch.set_num_threads(1)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tr.init(opt)
vocab = tr.loadVocab(opt.task)
ivocab = tr.loadiVocab(opt.task)
opt.numWords = len(ivocab)
print("Vocal size: %s" % opt.numWords)
print('loading data ..')

train_dataset = tr.loadData('train', opt.task)
print("Size of training data: %s" % len(train_dataset))
dev_dataset = tr.loadData('dev', opt.task)
print("Size of dev data: %s" % len(dev_dataset))
test_dataset = tr.loadData('test',opt.task)

model = compAggWikiqa(opt).cuda()

# pretrain
cur_best_MAP = 0.714
for i in range(opt.pretrain_epochs):
    random.shuffle(train_dataset)
    rl_pretrain(model, train_dataset,opt,i)
    results = model.predict_dataset(train_dataset,dev_dataset,test_dataset,rl_predict=opt.rl_pretrain)
    for j, vals in enumerate(results):
        if j == 0:
            print("Dev: MAP: %s, MRR: %s" % (vals[0], vals[1]))
        else:
            print("Test: MAP: %s, MRR: %s" % (vals[0], vals[1]))
            test_MAP = vals[0]
    if test_MAP > cur_best_MAP:
        print('saving pretrained parameters')
        print()
        torch.save(model.state_dict(),'pos_params.pkl')
        cur_best_MAP = test_MAP

print('loading pretrained parameters...')
model.load_state_dict(torch.load('params.pkl'))
results = model.predict_dataset(train_dataset,dev_dataset,test_dataset,rl_predict=opt.rl_pretrain)
for j, vals in enumerate(results):
    if j == 0:
        print("pretrained Dev: MAP: %s, MRR: %s" % (vals[0], vals[1]))
    else:
        print("pretrained Test: MAP: %s, MRR: %s" % (vals[0], vals[1]))


# RL training
for i in range(opt.max_epochs):
    # random.shuffle(train_dataset)
    train(model, train_dataset,opt,i)
    model.optim_state['learningRate'] = model.optim_state['learningRate'] * opt.lr_decay
    results = model.predict_dataset(train_dataset,dev_dataset,test_dataset,rl_predict=True)
    model.save('../trainedmodel/', opt, results, i)
