import torch
import torch.nn as nn



class CapsuleLayer(nn.Module):
    def __init__(self, args):
        super(CapsuleLayer, self).__init__()
        # input vector num
        self.num_route_nodes = 1
        self.num_iterations = 3
        # vectors of hidden capsules
        self.num_hidden = args.hidden_capsule
        self.route_weights1 = nn.Parameter(torch.randn(self.num_hidden, args.feature_dim, args.feature_dim))


        # output vector num
        self.num_capsules = args.center_num
        self.gpu = args.gpu
        self.route_weights = nn.Parameter(torch.randn(self.num_capsules, args.feature_dim, args.feature_dim))

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            # layer 1:
            priors1 = torch.matmul(x[None, :, :, None, :], self.route_weights1[:, None, None, :, :])
            if torch.cuda.is_available():
                logits1 = torch.autograd.Variable(torch.zeros(priors1.size())).cuda(self.gpu)
            else:
                logits1 = torch.autograd.Variable(torch.zeros(priors1.size()))
            for i in range(self.num_iterations):
                # print('logits', logits.shape)
                probs1 = nn.Softmax(logits1, dim=2)
                outputs1 = self.squash((torch.mul(probs1, priors1)).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits1 = (torch.mul(priors1, outputs1)).sum(dim=-1, keepdim=True)
                    logits1 = logits1 + delta_logits1

            outputs1 = outputs1.squeeze()
            outputs1 = torch.transpose(outputs1, 0, 1).contiguous()

            # print('out put 1', outputs1.shape)

            priors = torch.matmul(outputs1[None, :, :, None, :], self.route_weights[:, None, None, :, :])
            if torch.cuda.is_available():
                logits = torch.autograd.Variable(torch.zeros(priors.size())).cuda(self.gpu)
            else:
                logits = torch.autograd.Variable(torch.zeros(priors.size()))
            for i in range(self.num_iterations):
                # print('logits', logits.shape)
                probs = nn.Softmax(logits, dim=2)
                outputs = self.squash((torch.mul(probs, priors)).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (torch.mul(priors, outputs)).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
            # print('outputs: ', outputs.shape)
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)
        outputs = outputs.squeeze()
        # print('out put ', outputs.shape)
        return outputs