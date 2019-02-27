import numpy as np
import torch
from torch.autograd import Variable
from compAggCNN import compAggWikiqa
from metrics import MAP


# supervised training
def super_pretrain(model: compAggWikiqa, dataset: list,opt,current_epoch):
    model.proj_modules.train()
    model.dropout_modules.train()
    model.emb_vecs.train()
    model.conv_module.train()

    dataset_size = len(dataset)
    indices = [x for x in range(len(dataset))]

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    for i in range(0, dataset_size, model.batch_size):  # TODO change me for debug: # 1, model.batch_size):
        batch_size = min(model.batch_size,
                         dataset_size - i)  # min(i + model.batch_size - 1, dataset_size) - i + 1  # TODO: why?
        loss = 0.
        for j in range(0, batch_size):
            idx = indices[i + j]
            data_raw = dataset[idx]
            data_q, data_as, data_as_len, label = data_raw
            data_q = data_q.cuda()
            data_as = data_as.cuda()
            data_as_len = torch.cuda.LongTensor(data_as_len)
            label = Variable(label, requires_grad=False).cuda()
            _,soft_output = model.comp_agg(data_q, data_as, data_as_len)
            example_loss = model.criterion(soft_output, label)
            loss += example_loss
        loss = loss / batch_size
        if (i / opt.batch_size) % 10 == 0:
            print('supervised pretrain epoch %d, step %d, loss' % (current_epoch + 1, i / opt.batch_size), loss.data.cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


#train
def train(model: compAggWikiqa,dataset: list,opt,current_epoch,mode):
    model.proj_modules.train()
    model.dropout_modules.train()
    model.emb_vecs.train()
    model.conv_module.train()
    dataset_size = len(dataset)
    indices = [x for x in range(len(dataset))]
    # indices = [667] + [x for x in range(667)] + [x for x in range(668, 873)]  # TODO: remove me
    # for i in range(len(dataset)):
    #     print(i)
    #     print(dataset[i][-1])

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr*(opt.lr_decay**current_epoch))

    for i in range(0, dataset_size, model.batch_size):  # TODO change me for debug: # 1, model.batch_size):
        batch_size = min(model.batch_size,
                         dataset_size - i)  # min(i + model.batch_size - 1, dataset_size) - i + 1  # TODO: why?
        loss = 0.

        for j in range(0, batch_size):
            idx = indices[i + j]
            # print(idx)
            data_raw = dataset[idx]
            data_q, data_as,data_as_len, label = data_raw
            data_q = data_q.cuda()
            data_as = data_as.cuda()
            data_as_len = torch.cuda.LongTensor(data_as_len)
            label = Variable(label, requires_grad=False).cuda()
            # get predicted results
            new_score = model(data_q, data_as,data_as_len)
            if mode=='pretrain':
                example_loss = model.criterion(new_score, label)
                loss += example_loss
            else:
                # calculate the rewards
                conf = new_score.data.cpu().numpy() # conf -> confidence -> 置信度'
                rewards = np.full((len(conf)),1.)
                previous_map = 1.
                for k in range(1,len(conf)+1):
                    top_i_th_conf = new_score[:k].data
                    top_i_th_gt = label.data[:k]
                    cur_map = MAP(top_i_th_gt,top_i_th_conf)  # current MAP
                    rewards[k-1] = rewards[k-1] + previous_map - cur_map
                    previous_map = cur_map
                rewards = torch.cuda.FloatTensor(rewards)
                example_loss = 0.
                for k in range(len(label)):
                    # example_loss += rewards[k]*new_score[k]
                    example_loss += rewards[k]*model.criterion(new_score[k], label[k])
                loss += example_loss
        loss = loss / batch_size
        if (i/opt.batch_size)%10 == 0:
            print('train epoch %d, step %d, loss %f, lr %f'
                   %(current_epoch+1,
                     i/opt.batch_size,
                     loss.data.cpu().numpy()[0],
                     opt.lr*(opt.lr_decay**current_epoch)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
