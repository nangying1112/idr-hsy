import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

# data = torch.load('data/wikiqa/sequence/test_py.t7')
# for i in range(100):
#
#     print(data[i][2])
as_tmp = torch.LongTensor(5).fill_(0)
print(as_tmp)
a = [[1.,1.,1.]]
a = torch.FloatTensor(a)

b = [[1.,5.,1.]]
b = torch.FloatTensor(b)
d = []
d.append(a)
d.append(b)
c = torch.cat([a,b],0)
# c = c[:, 1]
print(c)


class NewProjModule(nn.Module):
    def __init__(self, emb_dim, mem_dim):
        super(NewProjModule, self).__init__()
        self.emb_dim = emb_dim
        self.mem_dim = mem_dim
        self.linear1 = nn.Linear(self.emb_dim, self.mem_dim)
        self.linear2 = nn.Linear(self.emb_dim, self.mem_dim)

    def forward(self, input):
        i = nn.Sigmoid()(self.linear1(input))
        print(i)
        u = nn.Tanh()(self.linear2(input))
        print(u)
        out = i.mul(u)  # CMulTable().updateOutput([i, u])
        return out

proj = NewProjModule(3,2)
input = Variable(c)
rinput = torch.FloatTensor([[1.,2.,3.],[2.,3.,4.]])
rinput = Variable(rinput)

lPad = input.view(-1, input.size(0), input.size(1))
print(lPad)
lPad = input  # self.lPad = Padding(0, 0)(linput) TODO: figureout why padding?
print(lPad)

M_r = torch.mm(lPad, rinput.t())
print(rinput.t())
alpha = F.softmax(M_r.transpose(0, 1))
print(alpha)
Yl = torch.mm(alpha, lPad)
print(Yl)
print(torch.max(Yl))
print(input.mul(rinput))


conf = [2,3,1,-0.1,0.2]
# conf = Variable(conf)
# out = F.log_softmax(conf)
print(conf)


# print(np.random.rand(len(conf)))
# print(np.random.rand(len(conf)) / conf)
# print((np.floor(np.random.rand(len(conf)) / conf)))
# print(np.clip([ 0.,0.5,2.,-3.,4.], 1, 0))
lu = np.clip(np.floor(np.random.rand(len(conf)) / conf), 1, 0).astype(int)
print(lu)
#
#
# label = torch.FloatTensor([1.,0.,0.])
# conf = torch.FloatTensor([0.332,0.333,0.334])
# conf = conf[:1]
# print(conf)
# # for i in range(1,3):
# #     res = MAP(label[:i],conf[:i])
# #     print(res)
# #     print()
#
# for i in range(1,2):
#     print(i)
for i in range(2,2):
    print(i)


