import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gat import GraphAttentionNet


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, src, positive, negative):
        cos_sim_pos = F.cosine_similarity(src, positive, dim=1)
        cos_sim_neg = F.cosine_similarity(src, negative, dim=1)

        n_neg_sampling = int(cos_sim_neg.shape[0] // cos_sim_pos.shape[0])
        cos_sim_pos = torch.repeat_interleave(cos_sim_pos, n_neg_sampling, 0)
        loss = torch.mean(torch.relu(cos_sim_neg - cos_sim_pos + self.margin))
        return loss

class GraphModel(nn.Module):

    def __init__(self, input_dim, out_dim, num_head=1, dropout=0.1, margin=1, alpha=0.1, beta=0.1, lamda=0.9):
        super(GraphModel, self).__init__()
        self.model = GraphAttentionNet(input_dim, out_dim, num_head=num_head, dropout=dropout)
        self.linkloss = TripletLoss(margin=margin)
        
    def cal_cos_simlarity(self, embedding1, embedding2):
        return torch.cosine_similarity(embedding1, embedding2, dim=1).view(-1)

    def computeLoss(self, x, adj, src_index, dst_pos_index, dst_neg_index):
        
        embeddings = self.model(x, adj)
        src_embeddings = embeddings[src_index, :]
        dst_pos_embeddings = embeddings[dst_pos_index, :]
        dst_neg_embeddings = embeddings[dst_neg_index, :]
        linkloss = self.linkloss(src_embeddings, dst_pos_embeddings, dst_neg_embeddings)

        return linkloss


    def inference(self, x, adj, pairs, thre_cos=0.5):
        
        src = pairs[:, 0]
        dest = pairs[:, 1]
        embeddings = self.model(x, adj)
        src_embeddings = embeddings[src, :]
        dest_embeddings = embeddings[dest, :]
        simlarity = self.cal_cos_simlarity(src_embeddings, dest_embeddings)

        pred = torch.zeros(simlarity.shape, dtype=torch.int)
        pred[simlarity > thre_cos] = 1
        simlarity[simlarity < 0] = 0
        return pred, simlarity

