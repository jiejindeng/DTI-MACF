import torch
import random
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F



class combiner(nn.Module):
    def __init__(self, embedding1, embedding2, embedding3, embedding4, embedding5, embedding_dim, droprate, cuda = 'cpu'):
        super(combiner, self).__init__()

        self.embedding1 = embedding1
        self.embedding2 = embedding2
        self.embedding3 = embedding3
        ##########################
        self.embedding4 = embedding4
        self.embedding5 = embedding5
        ##########################
        self.embed_dim = embedding_dim
        self.droprate = droprate
        self.device = cuda

        # self.att1 = nn.Linear(self.embed_dim * 3, self.embed_dim)
        # self.att2 = nn.Linear(self.embed_dim, 3)
        self.att1 = nn.Linear(self.embed_dim *5, self.embed_dim)
        self.att2 = nn.Linear(self.embed_dim, 5)
        self.softmax = nn.Softmax()

    def forward(self, nodes_u, nodes_i):
        embedding1, nodes_fea1 = self.embedding1(nodes_u, nodes_i)
        embedding2, nodes_fea2 = self.embedding2(nodes_u, nodes_i)
        embedding3, nodes_fea3 = self.embedding3(nodes_u, nodes_i)
        embedding4, nodes_fea4 = self.embedding4(nodes_u, nodes_i)
        embedding5, nodes_fea5 = self.embedding5(nodes_u, nodes_i)

        # nodes_fea = torch.cat((nodes_fea1, nodes_fea2, nodes_fea3), dim=1)
        # x = torch.cat((embedding1, embedding2, embedding3), dim = 1)
        # nodes_fea = F.relu(self.att1(nodes_fea).to(self.device), inplace = True)
        # x = F.relu(self.att1(x).to(self.device), inplace = True)
        # nodes_fea = F.dropout(nodes_fea, training = self.training)
        # x = F.dropout(x, training = self.training)
        # nodes_fea = self.att2(nodes_fea).to(self.device)
        # x = self.att2(x).to(self.device)
        #
        # att_w = F.softmax(x, dim = 1)
        # att_n = F.softmax(nodes_fea, dim=1)
        # att_w1, att_w2, att_w3 = att_w.chunk(3, dim = 1)
        # att_n1, att_n2, att_n3 = att_n.chunk(3, dim = 1)
        # att_w1.repeat(self.embed_dim, 1)
        # att_w2.repeat(self.embed_dim, 1)
        # att_w3.repeat(self.embed_dim, 1)
        # att_n1.repeat(self.embed_dim, 1)
        # att_n2.repeat(self.embed_dim, 1)
        # att_n3.repeat(self.embed_dim, 1)
        #
        # final_embed_matrix = torch.mul(embedding1, att_w1) + torch.mul(embedding2, att_w2) + torch.mul(embedding3, att_w3)
        # final_nodes_fea = torch.mul(nodes_fea1, att_n1) + torch.mul(nodes_fea2, att_n2) + torch.mul(nodes_fea3, att_n3)
        # return final_embed_matrix, final_nodes_fea

        nodes_fea = torch.cat((nodes_fea1, nodes_fea2, nodes_fea3, nodes_fea4,nodes_fea5), dim=1)
        # nodes_fea = nodes_fea1
        x = torch.cat((embedding1, embedding2, embedding3, embedding4, embedding5), dim=1)
        # x = embedding1
        nodes_fea = F.relu(self.att1(nodes_fea).to(self.device), inplace=True)
        x = F.relu(self.att1(x).to(self.device), inplace=True)
        nodes_fea = F.dropout(nodes_fea, training=self.training)
        x = F.dropout(x, training=self.training)
        nodes_fea = self.att2(nodes_fea).to(self.device)
        x = self.att2(x).to(self.device)

        att_w = F.softmax(x, dim = 1)
        att_n = F.softmax(nodes_fea, dim=1)
        att_w1, att_w2, att_w3, att_w4, att_w5 = att_w.chunk(5, dim = 1)
        att_n1, att_n2, att_n3, att_n4, att_n5 = att_n.chunk(5, dim = 1)
        att_w1.repeat(self.embed_dim, 1)
        att_w2.repeat(self.embed_dim, 1)
        att_w3.repeat(self.embed_dim, 1)
        att_w4.repeat(self.embed_dim, 1)
        att_w5.repeat(self.embed_dim, 1)

        att_n1.repeat(self.embed_dim, 1)
        att_n2.repeat(self.embed_dim, 1)
        att_n3.repeat(self.embed_dim, 1)
        att_n4.repeat(self.embed_dim, 1)
        att_n5.repeat(self.embed_dim, 1)

        final_embed_matrix = torch.mul(embedding1, att_w1) + torch.mul(embedding2, att_w2) + torch.mul(embedding3, att_w3) +torch.mul(embedding4, att_w4) + torch.mul(embedding5,att_w5)
        final_nodes_fea = torch.mul(nodes_fea1, att_n1) + torch.mul(nodes_fea2, att_n2) + torch.mul(nodes_fea3, att_n3)+ torch.mul(nodes_fea4, att_n4) + torch.mul(nodes_fea5, att_n5)
        return final_embed_matrix, final_nodes_fea