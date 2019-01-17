import numpy as np
import torch
import copy
import torch.nn as nn
from wikiqa.compAggWikiqa import compAggWikiqa

class RL(nn.Module):
    def __init__(self,opt):
        super(RL, self).__init__()
        self.qa_CompAgg = compAggWikiqa(opt).cuda()
        self.neg_CompAgg = compAggWikiqa(opt).cuda()
        self.pos_CompAgg = compAggWikiqa(opt).cuda()

    def forward(self,data_q,data_as,label):
        q_a_state, q_a_score = self.qa_CompAgg(data_q, data_as) # initially modeling the question and all answers
        states = []
        for i in range(len(data_as)):
            q_a_score_np = q_a_score.data.cpu().numpy()
            neg_a_index = np.argmin(q_a_score_np[0:i]) # get the index of the most negative answer predicted in this step
            pos_a_index = np.argmax(q_a_score_np[0:i]) # positive
            neg_a = [data_as[neg_a_index]]
            pos_a = [data_as[pos_a_index]]
            cur_neg_state, _ = self.neg_CompAgg(data_as[i],neg_a)
            cur_pos_state, _ = self.pos_CompAgg(data_as[i],pos_a)
            tmp_state = torch.cat([q_a_state,cur_neg_state,cur_pos_state],1)
            states.append(tmp_state)
        _, soft_output = self.qa_CompAgg(data_q,states)


















