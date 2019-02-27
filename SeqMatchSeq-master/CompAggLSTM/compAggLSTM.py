import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy

import loadFiles as tr
from DMax import DMax



class compAggWikiqa(nn.Module):

    def __init__(self, args):
        super(compAggWikiqa, self).__init__()
        self.mem_dim = args.mem_dim
        # self.att_dim = args.att_dim
        # self.cov_dim = args.cov_dim
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.emb_dim = args.wvecDim
        self.task = args.task
        self.numWords = args.numWords
        self.dropoutP = args.dropoutP
        self.grad = args.grad
        self.visualize = args.visualize
        self.emb_lr = args.emb_lr
        self.emb_partial = args.emb_partial
        self.comp_type = args.comp_type
        self.window_sizes = args.window_sizes
        self.window_large = args.window_large
        self.gpu = args.gpu
        self.best_score = 0

        self.emb_vecs = nn.Embedding(self.numWords, self.emb_dim)
        self.emb_vecs.weight.data = tr.loadVacab2Emb(self.task)

        self.att_module_master = self.new_att_module()
        if self.comp_type == "mul":
            self.sim_sg_module = self.new_sim_mul_module()
        else:
            Exception("The word matching method is not provided")

        # self.conv_module = self.new_conv_module()
        self.gru_module = self.bi_GRU(args)
        mem_dim = self.mem_dim

        class TempNet(nn.Module):
            def __init__(self, dim):
                super(TempNet, self).__init__()
                self.layer1 = nn.Linear(dim, 1)
                # self.layer2 = nn.Linear(150,1)
                # self.tanh = nn.Tanh()

            def forward(self, input):
                var1 = self.layer1(input)
                var1 = var1.view(-1)
                # print('var1',var1)
                out = F.log_softmax(var1)
                # print(out)
                return out


        self.soft_module = TempNet(2*args.gru_dim)
        self.rl_soft_module = TempNet(4*args.gru_dim)
        self.optim_state = {"learningRate": self.learning_rate}
        self.criterion = nn.KLDivLoss()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout_modules = [None] * 2
        self.proj_modules = [None] * 2
        self.proj_modules = self.new_proj_module()
        self.dropout_modules = nn.Dropout(self.dropoutP)
        # for aa comp_agg
        self.aa_proj_modules = self.new_proj_module()
        self.aa_att_module_master = self.new_att_module()
        self.aa_gru_module = self.bi_GRU(args)

        self.aa_soft_module = TempNet(2*args.gru_dim)
        self.cur_trans = nn.Linear(mem_dim,75)
        self.state_trans = nn.Linear(mem_dim,75)
        # self.cos = nn.CosineSimilarity(dim=1,eps=1e-6)

    def new_proj_module(self):
        emb_dim = self.emb_dim
        mem_dim = self.mem_dim

        class NewProjModule(nn.Module):
            def __init__(self, emb_dim, mem_dim):
                super(NewProjModule, self).__init__()
                self.emb_dim = emb_dim
                self.mem_dim = mem_dim
                self.linear1 = nn.Linear(self.emb_dim, self.mem_dim)
                self.linear2 = nn.Linear(self.emb_dim, self.mem_dim)

            def forward(self, input):
                # why project two times??
                if len(input.size()) == 3:
                    b_size = input.size()[0]
                    input = input.view(-1,input.size()[-1])
                    i = nn.Sigmoid()(self.linear1(input))
                    u = nn.Tanh()(self.linear2(input))
                    out = i.mul(u)
                    out = out.view(b_size, -1, out.size()[-1])
                else:
                    i = nn.Sigmoid()(self.linear1(input))
                    u = nn.Tanh()(self.linear2(input))
                    out = i.mul(u)
                return out

        module = NewProjModule(emb_dim, mem_dim)
        return module

    def new_att_module(self):

        class NewAttModule(nn.Module):
            def __init__(self):
                super(NewAttModule, self).__init__()

            def forward(self, linput, rinput): # linput:question, rinput:answer

                # self.lPad = linput.view(-1, linput.size(0), linput.size(1))
                self.lPad = linput  # self.lPad = Padding(0, 0)(linput) TODO: figureout why padding?
                if len(self.lPad.size()) == 3:
                    # self.lPad = self.lPad.permute(0,2,1)
                    # print('lpad',self.lPad.size())
                    # print('rinput',rinput.permute(0,2,1).size())
                    self.M_r = torch.bmm(self.lPad, rinput.permute(0,2,1))

                    self.alpha = F.softmax(self.M_r.permute(0,2,1))
                    self.Yl = torch.bmm(self.alpha, self.lPad)
                else:
                    # print('lpad',self.lPad.size())
                    b_size = rinput.size()[0]
                    rinput = rinput.view(-1,rinput.size()[2])
                    self.M_r = torch.mm(self.lPad, rinput.t())
                    self.alpha = F.softmax(self.M_r.transpose(0, 1))
                    self.Yl = torch.mm(self.alpha, self.lPad)
                    self.Yl = self.Yl.view(b_size,-1,self.Yl.size()[1])
                    # print('yi',self.Yl.size())
                return self.Yl

        att_module = NewAttModule()
        if getattr(self, "att_module_master", None):
            for (tar_param, src_param) in zip(att_module.parameters(), self.att_module_master.parameters()):
                tar_param.grad.data = src_param.grad.data.clone()
        return att_module

    def new_conv_module(self):
        window_sizes = self.window_sizes
        cov_dim = self.cov_dim
        mem_dim = self.mem_dim
        gpu = self.gpu

        class TemporalConvoluation(nn.Module):
            def __init__(self, cov_dim, mem_dim, window_size):
                super(TemporalConvoluation, self).__init__()
                self.conv1 = nn.Conv1d(in_channels=mem_dim, out_channels=cov_dim, kernel_size=window_size).cuda()

            def forward(self, input):
                input = input.permute(0,2,1)
                # myinput = input.view(1, input.size()[0], input.size()[1]).transpose(1, 2)  # 1, 150, 56
                output = self.conv1(input).permute(0,2,1)  # 56, 150
                return output

        # class MyTemporalConvoluation2(nn.Module):
        #     def __init__(self, cov_dim, mem_dim, window_size):
        #         super(MyTemporalConvoluation2, self).__init__()
        #         self.inp, self.outp, self.kw, self.dw = cov_dim, mem_dim, window_size, 1
        #         self.weight = Variable(torch.cuda.Tensor(self.outp, self.inp * self.kw), requires_grad=False)
        #         self.bias = Variable(torch.cuda.Tensor(self.outp), requires_grad=False)
        #         self.reset()
        #
        #     def reset(self, stdv=None):
        #         if stdv is not None:
        #             stdv = stdv * math.sqrt(3)
        #         else:
        #             stdv = 1. / math.sqrt(self.kw * self.inp)
        #
        #         self.weight.data.uniform_(-stdv, stdv)
        #         self.bias.data.uniform_(-stdv, stdv)
        #
        #     def forward(self, input):
        #         weights = self.weight.view(-1, self.inp)  # weights applied to all
        #         bias = self.bias
        #         nOutputFrame = int((input.size(0) - self.kw) / self.dw + 1)
        #
        #         output = Variable(torch.cuda.FloatTensor(nOutputFrame, self.outp))
        #
        #         for i in range(input.size(0)):  # do -- for each sequence element
        #             element = input[i]  # ; -- features of ith sequence element
        #             output[i] = element.mm(weights) + bias
        #         return output

        class NewConvModule(nn.Module):
            def __init__(self, window_sizes, cov_dim, mem_dim):
                super(NewConvModule, self).__init__()
                self.window_sizes = window_sizes
                self.cov_dim = cov_dim
                self.mem_dim = mem_dim

                self.d_tempconv = {}
                self.d_dmax = {}
                for window_size in self.window_sizes:
                    self.d_tempconv[window_size] = TemporalConvoluation(self.cov_dim, self.mem_dim, window_size)
                    self.d_dmax[window_size] = DMax(dimension=0, windowSize=window_size, gpu=gpu)
                self.linear1 = nn.Linear(len(window_sizes) * cov_dim, cov_dim)
                self.relu1 = nn.ReLU()
                self.tanh1 = nn.Tanh()

            def forward(self, input, sizes):
                conv = [None] * len(self.window_sizes)
                pool = [None] * len(self.window_sizes)
                for i, window_size in enumerate(self.window_sizes):
                    tempconv = self.d_tempconv[window_size](input)
                    conv[i] = self.relu1(tempconv)
                    # print('conv',conv[i].size())
                    # pool[i] = self.d_dmax[window_size](conv[i], sizes)
                    pool[i] = torch.squeeze(torch.max(conv[i],1)[0]) # max返回值有两个，一个是最大值，一个是最大值索引
                    if len(pool[i].size())==1:
                        pool[i] = torch.unsqueeze(pool[i],0)
                    # print('pool',pool[i].size())
                # print('tempconv',type(tempconv.data))
                # print('pool',type(conv[i].data))
                # print('pool',type(pool[i].data))
                concate = torch.cat(pool, 1)  # JoinTable(2).updateOutput(pool)
                # print('concate',concate.size())
                linear1 = self.linear1(concate)
                # print('linear1',type(linear1.data))
                output = self.tanh1(linear1)
                # print('output',output)
                return output

        conv_module = NewConvModule(window_sizes, cov_dim, mem_dim)
        return conv_module

    def bi_GRU(self,args):
        class BiGRU(nn.Module):
            def __init__(self, args):
                super(BiGRU, self).__init__()
                self.args = args
                self.input_size = args.mem_dim
                self.hidden_dim = args.gru_dim
                self.num_layers = args.gru_layers
                # C = args.class_num
                # gru
                self.bigru = nn.GRU(self.input_size,
                                    self.hidden_dim,
                                    dropout=args.dropoutP,
                                    num_layers=self.num_layers,
                                    bidirectional=True)
                # linear
                # self.hidden2label = nn.Linear(self.hidden_dim * 2, C)
                #  dropout
                self.dropout = nn.Dropout(args.dropoutP)

            def forward(self, input):
                # gru
                gru_out, _ = self.bigru(input)
                # print('gru_output',gru_out.size())

                # gru_out = torch.transpose(gru_out, 0, 1)
                gru_out = torch.transpose(gru_out, 1, 2)
                # print('gru_output', gru_out.size())
                # pooling
                # gru_out = F.tanh(gru_out)
                gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
                # print('gru_output',gru_out.size())

                gru_out = F.tanh(gru_out)
                # linear
                # y = self.hidden2label(gru_out)
                # logit = y
                return gru_out
        gru_module = BiGRU(args=args)
        return gru_module

    def new_sim_mul_module(self):
        class NewSimMulModule(nn.Module):
            def __init__(self):
                super(NewSimMulModule, self).__init__()

            def forward(self, inputa, inputh):  # actually it's a_j vs h_j, element-wise mul
                return inputa.mul(inputh)  # return CMulTable().updateOutput([inputq, inputa])

        return NewSimMulModule()

    def comp_agg(self,data_q,data_as,data_as_len):
        for k in range(len(data_as)):
            if data_as_len[k] < self.window_large:
                data_as_len[k] = self.window_large
        inputs_a_emb = self.emb_vecs.forward(
            Variable(data_as.type(torch.cuda.LongTensor),
                     requires_grad=False))  # TODO: why LongTensor would convert to Float
        inputs_q_emb = self.emb_vecs.forward(Variable(data_q, requires_grad=False))

        inputs_a_emb = self.dropout_modules.forward(inputs_a_emb)
        inputs_q_emb = self.dropout_modules.forward(inputs_q_emb)
        projs_a_emb = self.proj_modules.forward(inputs_a_emb)
        projs_q_emb = self.proj_modules.forward(inputs_q_emb)

        if data_q.size()[0] == 1:
            projs_q_emb = projs_q_emb.resize(1, self.mem_dim)
        # question-awared answer representation
        att_output = self.att_module_master.forward(projs_q_emb, projs_a_emb)
        # print('att_output',att_output.size())
        sim_output = self.sim_sg_module.forward(projs_a_emb, att_output)
        # print('sim',sim_output.size())
        # conv_output = self.conv_module.forward(sim_output, data_as_len)
        gru_output = self.gru_module(sim_output)
        soft_output = self.soft_module.forward(gru_output)
        return gru_output, soft_output

    def aa_comp_agg(self,data_q,data_as,data_as_len):
        for k in range(len(data_as)):
            data_as_len[k] = data_as[k].size()[0]
            if data_as_len[k] < self.window_large:
                data_as_len[k] = self.window_large
        inputs_a_emb = self.emb_vecs.forward(
            Variable(data_as.type(torch.cuda.LongTensor),
                     requires_grad=False))  # TODO: why LongTensor would convert to Float
        inputs_q_emb = self.emb_vecs.forward(Variable(data_q, requires_grad=False))

        inputs_a_emb = self.dropout_modules.forward(inputs_a_emb)
        inputs_q_emb = self.dropout_modules.forward(inputs_q_emb)

        projs_a_emb = self.aa_proj_modules.forward(inputs_a_emb)
        projs_q_emb = self.aa_proj_modules.forward(inputs_q_emb)

        # if data_q.size()[0] == 1:
        #     projs_q_emb = projs_q_emb.resize(1, self.mem_dim)
        # question-awared answer representation
        att_output = self.aa_att_module_master.forward(projs_q_emb, projs_a_emb)
        # print('att_output',att_output.size())
        sim_output = self.sim_sg_module.forward(projs_a_emb, att_output)
        # print('sim_output',sim_output.size())
        gru_output = self.aa_gru_module.forward(sim_output)
        # print('gru_output',gru_output.size())
        soft_output = self.aa_soft_module.forward(gru_output)
        return gru_output, soft_output

    def rl_state(self,data_q,data_as,data_as_len,q_a_state, q_a_score):

        q_a_score_np = q_a_score.data.cpu().numpy()
        '''
        # concat the origin texts simply 
        rl_state = model.cur_trans(q_a_state[0].view(1, q_a_state[0].size(0)))
        for k in range(2, len(data_as) + 1):
            pos_a_index = np.argmax(q_a_score_np[0:k])  # positive
            q_cur_state = q_a_state[k - 1].view(1, q_a_state[0].size(0))
            if pos_a_index == k - 1:
                rl_state = torch.cat([rl_state, model.cur_trans(q_cur_state)], 0)
                # print('rl_state',rl_state.size())
            else:
                q_pos_cat = torch.cat([data_q, data_as[pos_a_index]])
                current_answer = []
                current_answer.append(data_as[k - 1])
                q_pos_cur_state, _ = model.forward(q_pos_cat, current_answer)
                rl_state = torch.cat([rl_state, model.state_trans(q_pos_cur_state.view(1, q_a_state[0].size(0)))])
        rl_state = model.tanh(rl_state)
        '''
        # for the first answer
        overall_pos_index = np.argmax(q_a_score_np)
        reference = data_as[overall_pos_index].view(1, data_as.size()[1])
        # print('ref',reference.size())
        # print('data_q',data_q.size())
        # ref_info, _ = self.comp_agg(ref_answer, current_answer,data_as_len)
        # ref_info = ref_info.view(1, q_a_state[0].size(0))
        qa_info = q_a_state[0].view(1, q_a_state[0].size(0))
        for k in range(2, len(data_as) + 1):
            pos_a_index = np.argmax(q_a_score_np[0:k])  # positive
            cur_qa_info = q_a_state[k - 1].view(1, q_a_state[0].size(0))
            if pos_a_index == k - 1:
                cur_reference = data_as[overall_pos_index].view(1,data_as.size()[1])
                reference = torch.cat([reference,cur_reference],0)
                qa_info = torch.cat([qa_info, cur_qa_info], 0)
            else:
                cur_reference = data_as[overall_pos_index].view(1,data_as.size()[1])
                # cur_ref_info, _ = self.aa_comp_agg(ref_answer, current_answer,data_as_len)
                qa_info = torch.cat([qa_info, cur_qa_info], 0)
                reference = torch.cat([reference, cur_reference], 0)
        # rl_state = qa_info + ref_info  # add
        # print('refer',reference.size())
        # print('answer',data_as.size())
        ref_info,_ = self.aa_comp_agg(reference,data_as,data_as_len)
        # print(ref_info.size())
        rl_state = torch.cat([qa_info, ref_info], 1)
        # rl_state = self.relu(self.cur_trans(qa_info) + self.state_trans(ref_info))
        return rl_state

    def forward(self, data_q, data_as,data_as_len):
        # Prepare the data
        for k in range(len(data_as)):
        #     data_as_len[k] = data_as[k].size()[0]
        #     # Set the answer with a length less than 5 to [0,0,0,0,0]
        #     if data_as_len[k] < self.window_large:
                data_as_len[k] = self.window_large
        # compare-aggregate encode
        gru_output,soft_output = self.comp_agg(data_q,data_as,data_as_len)
        # get rl state
        rl_state = self.rl_state(data_q,data_as,data_as_len,gru_output,soft_output)
        new_score = self.rl_soft_module.forward(rl_state)
        return new_score

    def save(self, path, config, result, epoch):
        # print(result)
        assert os.path.isdir(path)
        recPath = path + config.task + str(config.expIdx) + 'Record.txt'

        file = open(recPath, 'a')
        if epoch == 0:
            for name, val in vars(config).items():
                file.write(name + '\t' + str(val) + '\n')
        file.write(config.task + ': ' + str(epoch) + ': ')
        for i, vals in enumerate(result):
            for _, val in enumerate(vals):
                file.write('%s, ' % val)
            # if i == 0:
            #     print("Dev: MAP: %s, MRR: %s" % (vals[0], vals[1]))
            # elif i == 1:
            #     print("Test: MAP: %s, MRR: %s" % (vals[0], vals[1]))
        print()
        file.write('\n')
        file.close()




