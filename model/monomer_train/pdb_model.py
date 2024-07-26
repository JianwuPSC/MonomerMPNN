from __future__ import print_function
import json, time, os, sys, glob
import shutil
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import torch.utils
import torch.utils.checkpoint

import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import itertools

def loss_nll(S, log_probs, mask):
    """ Negative log probabilities """ # 负对数似然
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size()) #[B*L, 21] [B*L]
    S_argmaxed = torch.argmax(log_probs,-1) #[B, L]
    true_false = (S == S_argmaxed).float() # 1/0
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av, true_false # loss[B,L] loss_av[B] true_false[B,L]


def loss_smoothed(S, log_probs, mask, weight=0.1):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(S, 21).float() # S [B,L,21]

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1)) # [B,L,21]+ 0.1/21
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True) # [B,L,21] / [B,L,21]

    loss = -(S_onehot * log_probs).sum(-1) # ([B,L,21]*[B,L,21]).sum(-1) -> [B,L]
    loss_av = torch.sum(loss * mask) / 2000.0 #fixed [B,L]*[B,L] ->[B]
    return loss, loss_av # loss[B,L], loss_av[B]


# The following gather functions
def gather_edges(edges, neighbor_idx): # edge feature
    # Features [B,L,L,C] at Neighbor indices [B,L,K] => Neighbor features [B,L,K,1]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors) #从输入张量中根据索引张量选择特定的元素 第2dim
    return edge_features

def gather_nodes(nodes, neighbor_idx): # node [B,L,C]   E_idx [B,L,K] # L*K 变相的attention
    # Features [B,L,L,C] at Neighbor indices [B,L,K] => Neighbor features [B,L,K,1]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1)) # [B,L,K] -> [B,LK]
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2)) # [B,LK,C]
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat) # [B,LK,C]
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1]) # [B,L,K,C]
    return neighbor_features # [B,L,K,C]

def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx): #node:[B,L,C] edge[B,L,K,C]  E_idx:[B,L,K]
    h_nodes = gather_nodes(h_nodes, E_idx) # [B,L,K,C]
    h_nn = torch.cat([h_neighbors, h_nodes], -1) # h_nn= h_V+h_E (node + edge)
    return h_nn # [B,L,K,2C]


class EncLayer(nn.Module):
    # h_EV:cat_neigh_node(h_V+h_E) -> message:act(linear(h_EV)) -> h_V:(h_V+message) -> h_EV:cat_neigh_node(h_V+h_E) -> message:act(linear(h_EV)) -> h_E:(h_E+message)
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None): # h_V[B,L,128], h_E[B,L,K,128], E_idx[B,L,K], mask_V[B,L], mask_atten[B,L,K]
        """ Parallel computation of full transformer layer """

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx) # h_EV [B,L,K,2C]
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1) # [B,L,K,C]
        h_EV = torch.cat([h_V_expand, h_EV], -1) # [B,L,K,3C]
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV))))) # h_message [B,L,K,128]
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message # [B,L,K,1]*[B,L,K,128]
        dh = torch.sum(h_message, -2) / self.scale # hidden [B,L,128]
        h_V = self.norm1(h_V + self.dropout1(dh)) # [B,L,128]

        dh = self.dense(h_V) # hidden [B,L,128]
        h_V = self.norm2(h_V + self.dropout2(dh)) #h_V=h_V + h_message [B,L,128]
        if mask_V is not None: # mask_V: mask
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V # [B,L,128]

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx) # h_EV [B,L,K,2C]
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message)) #h_E=h_E + h_message [B,L,128]
        return h_V, h_E # h_V[B,L,128], h_E[B,L,K,128]



class DecLayer(nn.Module): # h_EV:cat_neigh_node(h_V+h_E) -> message:act(linear(h_EV)) -> h_V:(h_V+message)
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None): # h_V[B,L,128], h_E[B,L,K,128], mask_V[B,L], mask_atten[B,L,K]
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1) # h_V[B,L,K,128]
        h_EV = torch.cat([h_V_expand, h_E], -1) # h_V_expand[B,L,K,128],h_E[B,L,K,128] -> h_EV[B,L,K,2C] 不再是cat_neighbors_nodes

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale # dh[B,L,2C]

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh)) #h_V = h_V + massage

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V # h_V [B,L,C]


class PositionWiseFeedForward(nn.Module): # act(linear)
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()
    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h

class PositionalEncodings(nn.Module): #编码 residue_index
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2*max_relative_feature+1+1, num_embeddings)

    def forward(self, offset, mask): # chain_encoding_all链的编码
        d = torch.clip(offset + self.max_relative_feature, 0, 2*self.max_relative_feature)*mask + (1-mask)*(2*self.max_relative_feature+1) # 超过clip值就是mask值，要预测的值
        d_onehot = torch.nn.functional.one_hot(d, 2*self.max_relative_feature+1+1) # one hot 编码特定维度 ->66 # index 用于onehot编码，offset.long()转换
        E = self.linear(d_onehot.float())# dim66 -> dim16
        return E


########## position embedding
class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, num_embeddings):
        super().__init__()

        # Compute the positional encodings once in log space.
        posi_embed = torch.zeros(max_len, d_model).float()
        posi_embed.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1) # [0, max_len, 1]
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        posi_embed[:, 0::2] = torch.sin(position * div_term)
        posi_embed[:, 1::2] = torch.cos(position * div_term)

        posi_embed = posi_embed.unsqueeze(0)
        self.register_buffer('posi_embed', posi_embed)

    def forward(self, offset, E_mask):
        position_embedding=self.posi_embed[:, :offset.size(1)]
        posi_mask = position_embedding * E_mask
        return posi_mask

###################

class ProteinFeatures(nn.Module): 
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, augment_eps=0., num_chain_embeddings=16):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        #self.embeddings = PositionalEmbedding(max_len=1000, num_positional_embeddings)

        node_in, edge_in = 6, num_positional_embeddings + num_rbf*25 # node_in:6, edge_in:16*25+16
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False) # edge embedding: position_embedding + rbf
        self.norm_edges = nn.LayerNorm(edge_features) # layerNorm

    def _dist(self, X, mask, eps=1E-6): # X [B,L,4,3] mask [B,L]
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2) # mask 2D [1,1,1] [B,L,L]
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2) # [B,1,L,4,3]-[B,L,1,4,3] -> [B,L,L,4,3]
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps) # [B,L,L] * [B,L,L,3]
        D_max, _ = torch.max(D, -1, keepdim=True) # [B,L,L,1]
        D_adjust = D + (1. - mask_2D) * D_max # [B,L,L,3]+ [B,L,L]*[B,L,L,1] 无错误，只用CA [B,L,3] 维度匹配
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False) # 返回最小元素及其索引
        return D_neighbors, E_idx # [B,L,K] [B,L,K]/[B,L,K]

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device) # 等差数列 2-22
        D_mu = D_mu.view([1,1,1,-1]) # [1,1,1,16]
        D_sigma = (D_max - D_min) / D_count  # 1.25
        D_expand = torch.unsqueeze(D, -1) # [B,L,L,3,1]
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2) # exp(-兰布达*|x-y|^2)
        return RBF #[B,L,L,3,16]

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K] -> [B,L,K,1] GNN gather
        RBF_A_B = self._rbf(D_A_B_neighbors) # rbf exp(-兰布达*|x-y|^2)  #[B,L,K,1,16]
        return RBF_A_B

    def forward(self, X, mask, aa_index, chain): # chain:chain_encoding_all
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X) # 加了个Gussion随机值
        
        b = X[:,:,1,:] - X[:,:,0,:] # CA - N
        c = X[:,:,2,:] - X[:,:,1,:] # C - CA
        a = torch.cross(b, c, dim=-1) # cross()
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:,:,1,:]
        Ca = X[:,:,1,:]
        N = X[:,:,0,:]
        C = X[:,:,2,:]
        O = X[:,:,3,:]
 
        D_neighbors, E_idx = self._dist(Ca, mask) # CA D_neighbor:[B,L,K,C]  E_idx:[B,L,K]

        RBF_all = [] # N,CA,C,O,Cb 5x5=25
        RBF_all.append(self._rbf(D_neighbors)) #Ca-Ca # [B,L,K,16] 两个点之间的距离高维找特征
        RBF_all.append(self._get_rbf(N, N, E_idx)) #N-N # [B,L,K,16] 两个点之间的距离高维找特征
        RBF_all.append(self._get_rbf(C, C, E_idx)) #C-C [B,L,K,16] 两个点之间的距离高维找特征
        RBF_all.append(self._get_rbf(O, O, E_idx)) #O-O [B,L,K,16] 两个点之间的距离高维找特征
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx)) #Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx)) #Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx)) #Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx)) #Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx)) #Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx)) #N-C
        RBF_all.append(self._get_rbf(N, O, E_idx)) #N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx)) #N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx)) #Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx)) #Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx)) #O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx)) #N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx)) #C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx)) #O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx)) #Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx)) #C-N
        RBF_all.append(self._get_rbf(O, N, E_idx)) #O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx)) #Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx)) #C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx)) #O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx)) #C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = aa_index[:,:,None]-aa_index[:,None,:] # [B,L,1]-[B,1,L] -> [B,L,L]
		# aa_index [[1,2,3,-100,-100],[100,200,300,-100,-100]] 既表征位置特征又表征链特征
        offset = gather_edges(offset[:,:,:,None], E_idx)[:,:,:,0] #[B,L,L,1] [B,L,K] -> [B,L,K,1]

        d_chains = ((chain[:, :, None] - chain[:,None,:])==0).long() #chain_encoding_all (0/1) # 链的编码信息[B,L,L] 整数
        E_chains = gather_edges(d_chains[:,:,:,None], E_idx)[:,:,:,0] # [B,L,K,1]
        E_positional = self.embeddings(offset.long(), E_chains) # position embedding （aa位置信息+链的信息）* 链的类别 [B,L,K,16]
        E = torch.cat((E_positional, RBF_all), -1) # posi_embedding + RBF
        E = self.edge_embedding(E) # [B,L,K,16] -> [B,L,K,128]
        E = self.norm_edges(E)
        return E, E_idx # E: edge[B,L,K,128]  E_idx:[B,L,K]



class ProteinMPNN(nn.Module):
    def __init__(self, num_letters=21, node_features=128, edge_features=128,
        hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
        vocab=21, k_neighbors=32, augment_eps=0.1, dropout=0.1):
        super(ProteinMPNN, self).__init__()

        # Hyperparameters
        self.node_features = node_features # 128
        self.edge_features = edge_features # 128
        self.hidden_dim = hidden_dim

        self.features = ProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps)
        #  E: edge[B,L,K,128]  E_idx:[B,L,K]
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim) # vocab_size=vocab, embedding_dim=hidden_dim

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim*3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, S, mask, ss_mask, aa_index, chain_encoding_all):
        """ Graph-conditioned sequence model """
        device=X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, aa_index, chain_encoding_all) # E: edge[B,L,K,128]  E_idx:[B,L,K]
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device) # h_V node [B,L,128]
        h_E = self.W_e(E) # h_E edge [B,L,K,128]

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1) #[B,L,K] (0/1 attention)
        mask_attend = mask.unsqueeze(-1) * mask_attend #[B,L,K]
        for layer in self.encoder_layers:
            h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend)
            # h_V[B,L,128], h_E[B,L,K,128] <-- # h_V[B,L,128], h_E[B,L,K,128], E_idx[B,L,K], mask_V[B,L], mask_atten[B,L,K]

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S) # seq_encode[B,L] 非 onehot
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx) #[B,L,C],[B,L,K,C],[B,L,K] -> [B,L,K,2C]

        # Build encoder embeddings [B,L,K,3C]
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx) #[B,L,C],[B,L,K,C],[B,L,K] -> [B,L,K,2C]
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx) #[B,L,C],[B,L,K,2C],[B,L,K] -> [B,L,K,3C]


        ss_mask = ss_mask*mask #update ss_mask to include missing regions # ss_mask:ss_maskask[B,L] 
        decoding_order = torch.argsort((ss_mask+0.0001)*(torch.abs(torch.randn(ss_mask.shape, device=device)))) #[B,L] 返回输入张量沿指定维度上元素的排序索引
        #[numbers will be smaller for places where ss_mask = 0.0 and higher for places where ss_mask = 1.0]
        
        mask_size = E_idx.shape[1] # L
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float() # [B,L,L] 0/1够稠密
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        # ij:[L,L]  biq:[B,L,L]  bjp:[B,L,L] -> bqp:[B,L,L] 输入张量中提取上三角部分。这个函数返回一个张量，其形状与输入张量相同，但只包含上三角部分的数据。
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1) # [B,L,K,1] **有链编码信息的,ss_mask 值越大,更分配一个attention
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1]) # [B,L,1,1]
        mask_bw = mask_1D * mask_attend # mask_backword [B,L,K,1] 1,predict
        mask_fw = mask_1D * (1. - mask_attend) # mask_forword [B,L,K,1] 0,fixed

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder # [B,L,K,3C]
        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx) # h_V + (h_E + h_S) -> h_ESV
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw # h_ESV + h_EXV_encoder(S_zero_like)
            h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_ESV, mask) #h_V[B,L,C]

        logits = self.W_out(h_V) # linear [B,L,21]
        log_probs = F.log_softmax(logits, dim=-1) # [B,L,21]
        return log_probs # [B,L,21]



class NoamOpt: # warpper step p[lr]=updata
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

def get_std_opt(parameters, d_model, step): #Adam optimizer
    return NoamOpt(
        d_model, 2, 4000, torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9), step)
