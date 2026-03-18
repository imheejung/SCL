# -*- coding: utf-8 -*-

import math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

import faiss
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class SCL(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset): 
        super(SCL, self).__init__(config, dataset)

        # load parameters info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.latent_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.smt_latent_dim = config['smt_latent_dim']

        self.ssl_temp = config['ssl_temp']
        self.ssl_reg = config['ssl_reg']
        self.hyper_layers = config['hyper_layers']

        self.alpha = config['alpha']

        self.reg_weight = config['reg_weight']
        self.potential_reg = config['potential_reg']
        self.semantic_reg = config['semantic_reg']
        self.k = config['num_clusters']

        # load semantic features info
        u_feat = np.load(config['user_feature_path'])
        i_feat = np.load(config['item_feature_path'])

        print('user feature shape:', u_feat.shape)
        print('item feature shape:', i_feat.shape)
        print('n_users:', self.n_users)
        print('n_items:', self.n_items)
        
        u_feat = torch.tensor(u_feat, dtype=torch.float).to(self.device)
        i_feat = torch.tensor(i_feat, dtype=torch.float).to(self.device)

        # projection layers for semantic features
        self.u_MLP = nn.Linear(u_feat.size(1), self.smt_latent_dim).to(self.device)
        self.i_MLP = nn.Linear(i_feat.size(1), self.smt_latent_dim).to(self.device)

        # define trainable embeddings
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        
        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        self.norm_adj_mat = self.get_norm_adj_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

        # semantic feature projection is fixed after initialization
        self.u_feat = self.u_MLP(u_feat).detach()
        self.i_feat = self.i_MLP(i_feat).detach()

        # freeze projection layers because semantic centers are fixed
        for p in self.u_MLP.parameters():
            p.requires_grad = False
        for p in self.i_MLP.parameters():
            p.requires_grad = False

        self.user_centroids = None
        self.user_2cluster = None
        self.item_centroids = None
        self.item_2cluster = None
        
        self.user_f_centroids = None
        self.user_f_2cluster = None
        self.item_f_centroids = None
        self.item_f_2cluster = None
        
    def semantic_step(self):
        
        self.user_f_centroids, self.user_f_2cluster = self.run_kmeans(self.u_feat.detach().cpu().numpy())
        self.item_f_centroids, self.item_f_2cluster = self.run_kmeans(self.i_feat.detach().cpu().numpy())        

    def e_step(self):
        
        user_embeddings = self.user_embedding.weight.detach().cpu().numpy()
        item_embeddings = self.item_embedding.weight.detach().cpu().numpy()

        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings)
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings)

    def run_kmeans(self, x):
        """Run K-means algorithm to get k clusters of the input tensor x
        """
        kmeans = faiss.Kmeans(d=self.latent_dim, k=self.k, gpu=True)
        kmeans.train(x)
        cluster_cents = kmeans.centroids

        _, I = kmeans.index.search(x, 1)

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(cluster_cents).to(self.device)
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = torch.LongTensor(I).squeeze().to(self.device)
        return centroids, node2cluster

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        for (row, col), value in data_dict.items():
            A[row, col] = value
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        self.diag = torch.from_numpy(diag).to(self.device)
        D = sp.diags(diag)
        L = D @ A @ D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        """Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """        
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for layer_idx in range(max(self.n_layers, self.hyper_layers*2)):
            all_embeddings = torch.sparse.mm(self.norm_adj_mat, all_embeddings)
            embeddings_list.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_list[:self.n_layers+1], dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings, embeddings_list

    def potential_loss(self, node_embedding, user, item):
        user_embeddings_all, item_embeddings_all = torch.split(node_embedding, [self.n_users, self.n_items])

        user_embeddings = user_embeddings_all[user]     # [B, e]
        norm_user_embeddings = F.normalize(user_embeddings)
        
        user2cluster = self.user_2cluster[user]     # [B,]
        user2centroids = self.user_centroids[user2cluster]   # [B, e]
        pos_score_user = torch.mul(norm_user_embeddings, user2centroids).sum(dim=1)
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.matmul(norm_user_embeddings, self.user_centroids.transpose(0, 1))
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        potential_nce_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        item_embeddings = item_embeddings_all[item]
        norm_item_embeddings = F.normalize(item_embeddings)

        item2cluster = self.item_2cluster[item]  # [B, ]
        item2centroids = self.item_centroids[item2cluster]  # [B, e]
        pos_score_item = torch.mul(norm_item_embeddings, item2centroids).sum(dim=1)
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.matmul(norm_item_embeddings, self.item_centroids.transpose(0, 1))
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        potential_nce_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        potential_nce_loss = self.potential_reg * (potential_nce_loss_user + self.alpha * potential_nce_loss_item)
        return potential_nce_loss

    def ssl_layer_loss(self, current_embedding, previous_embedding, user, item):
        current_user_embeddings, current_item_embeddings = torch.split(current_embedding, [self.n_users, self.n_items])
        previous_user_embeddings_all, previous_item_embeddings_all = torch.split(previous_embedding, [self.n_users, self.n_items])

        current_user_embeddings = current_user_embeddings[user]
        previous_user_embeddings = previous_user_embeddings_all[user]
        norm_user_emb1 = F.normalize(current_user_embeddings)
        norm_user_emb2 = F.normalize(previous_user_embeddings)
        norm_all_user_emb = F.normalize(previous_user_embeddings_all)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        current_item_embeddings = current_item_embeddings[item]
        previous_item_embeddings = previous_item_embeddings_all[item]
        norm_item_emb1 = F.normalize(current_item_embeddings)
        norm_item_emb2 = F.normalize(previous_item_embeddings)
        norm_all_item_emb = F.normalize(previous_item_embeddings_all)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)

        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        struct_loss = self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
        return struct_loss

    def semantic_loss(self, node_embedding, user, item):
        user_f_embeddings_all, item_f_embeddings_all = torch.split(node_embedding, [self.n_users, self.n_items])

        user_f_embeddings = user_f_embeddings_all[user]     # [B, e]
        norm_user_f_embeddings = F.normalize(user_f_embeddings)
        
        user2fcluster = self.user_f_2cluster[user]     # [B,]
        user2fcentroids = self.user_f_centroids[user2fcluster]   # [B, e]
        pos_score_user_f = torch.mul(norm_user_f_embeddings, user2fcentroids).sum(dim=1)
        pos_score_user_f = torch.exp(pos_score_user_f / self.ssl_temp)
        ttl_score_user_f = torch.matmul(norm_user_f_embeddings, self.user_f_centroids.transpose(0, 1))
        ttl_score_user_f = torch.exp(ttl_score_user_f / self.ssl_temp).sum(dim=1)

        smt_loss_user = -torch.log(pos_score_user_f / ttl_score_user_f).sum()

        item_f_embeddings = item_f_embeddings_all[item]
        norm_item_f_embeddings = F.normalize(item_f_embeddings)

        item2fcluster = self.item_f_2cluster[item]  # [B, ]
        item2fcentroids = self.item_f_centroids[item2fcluster]  # [B, e]
        pos_score_item_f = torch.mul(norm_item_f_embeddings, item2fcentroids).sum(dim=1)
        pos_score_item_f = torch.exp(pos_score_item_f / self.ssl_temp)
        ttl_score_item_f = torch.matmul(norm_item_f_embeddings, self.item_f_centroids.transpose(0, 1))
        ttl_score_item_f = torch.exp(ttl_score_item_f / self.ssl_temp).sum(dim=1)
        
        smt_loss_item = -torch.log(pos_score_item_f / ttl_score_item_f).sum()

        semantic_loss = self.semantic_reg * (smt_loss_user + self.alpha * smt_loss_item)
        return semantic_loss    
    
    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()

        center_embedding = embeddings_list[0] #user_embedding
        context_embedding = embeddings_list[self.hyper_layers * 2]

        struct_loss = self.ssl_layer_loss(context_embedding, center_embedding, user, pos_item)
        potential_loss = self.potential_loss(center_embedding, user, pos_item)
        
        semantic_loss = self.semantic_loss(center_embedding, user, pos_item)

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)

        mf_loss = self.mf_loss(pos_scores, neg_scores)

        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)

        return mf_loss + self.reg_weight * reg_loss, struct_loss, semantic_loss, potential_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, embedding_list = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
    
