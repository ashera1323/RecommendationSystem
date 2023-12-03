# -*- coding: utf-8 -*-

# !pip install torch_geometric
# !pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
# !pip install torch_sparse

# import required modules
import random
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import ctypes
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import model_selection, metrics, preprocessing
import copy
from torch_geometric.utils import degree

import torch
from torch import nn, optim, Tensor

from torch_sparse import SparseTensor, matmul

from torch_geometric.utils import structured_negative_sampling
from torch_geometric.data import download_url, extract_zip
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj

# defines LightGCN model
class LightGCN(MessagePassing):
    """LightGCN Model as proposed in https://arxiv.org/abs/2002.02126
    """

    def __init__(self, num_users,
                 num_items,
                 embedding_dim=64, # define the embding vector length for each node
                 K=3,
                 add_self_loops=False):
        """Initializes LightGCN Model

        Args:
            num_users (int): Number of users
            num_items (int): Number of items
            embedding_dim (int, optional): Dimensionality of embeddings. Defaults to 8.
            K (int, optional): Number of message passing layers. Defaults to 3.
            add_self_loops (bool, optional): Whether to add self loops for message passing. Defaults to False.
        """
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.K = K
        self.add_self_loops = add_self_loops

        # define user and item embedding for direct look up.
        # embedding dimension: num_user/num_item x embedding_dim

        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim) # e_u^0

        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim) # e_i^0

        # "Fills the input Tensor with values drawn from the normal distribution"
        # according to LightGCN paper, this gives better performance
        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

    def forward(self, edge_index: Tensor):
        """Forward propagation of LightGCN Model.

        Args:
            edge_index (SparseTensor): adjacency matrix

        Returns:
            tuple (Tensor): e_u_k, e_u_0, e_i_k, e_i_0
        """

        edge_index_norm = gcn_norm(edge_index=edge_index,
                                   add_self_loops=self.add_self_loops)

        # concat the user_emb and item_emb as the layer0 embing matrix
        # size will be (n_users + n_items) x emb_vector_len.   e.g: 10334 x 64
        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight]) # E^0

        embs = [emb_0] # save the layer0 emb to the embs list

        # emb_k is the emb that we are actually going to push it through the graph layers
        # as described in lightGCN paper formula 7
        emb_k = emb_0

        # push the embedding of all users and items through the Graph Model K times.
        # K here is the number of layers
        for i in range(self.K):
            emb_k = self.propagate(edge_index=edge_index_norm[0], x=emb_k, norm=edge_index_norm[1])
            embs.append(emb_k)


        # this is doing the formula8 in LightGCN paper

        # the stacked embs is a list of embedding matrix at each layer
        #    it's of shape n_nodes x (n_layers + 1) x emb_vector_len.
        #        e.g: torch.Size([10334, 4, 64])
        embs = torch.stack(embs, dim=1)

        # From LightGCn paper: "In our experiments, we find that setting α_k uniformly as 1/(K + 1)
        #    leads to good performance in general."
        emb_final = torch.mean(embs, dim=1) # E^K


        # splits into e_u^K and e_i^K
        users_emb_final, items_emb_final = torch.split(emb_final, [self.num_users, self.num_items])

        # returns e_u^K, e_u^0, e_i^K, e_i^0
        # here using .weight to get the tensor weights from n.Embedding
        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight

    def message(self, x_j, norm):
        # x_j is of shape:  edge_index_len x emb_vector_len
        #    e.g: torch.Size([77728, 64]
        #
        # x_j is basically the embedding of all the neighbors based on the src_list in coo edge index
        #
        # elementwise multiply by the symmetrically norm. So it's essentiall what formula 7 in LightGCN
        # paper does but here we are using edge_index rather than Adj Matrix
        return norm.view(-1, 1) * x_j

DEVICE = "cuda"  if torch.cuda.is_available() else 'cpu'
layers = 3
num_users = 943
num_movies = 1682


def convert_adj_mat_edge_index_to_r_mat_edge_index(input_edge_index):
    sparse_input_edge_index = SparseTensor(row=input_edge_index[0],
                                           col=input_edge_index[1],
                                           sparse_sizes=((num_users + num_movies), num_users + num_movies))
    adj_mat = sparse_input_edge_index.to_dense()
    interact_mat = adj_mat[: num_users, num_users :]
    r_mat_edge_index = interact_mat.to_sparse_coo().indices()
    return r_mat_edge_index

def get_user_positive_items(edge_index):
    """
    Generates dictionary of positive items for each user

    Args:
        edge_index (torch.Tensor): 2 by N list of edges

    Returns:
        dict: user -> list of positive items for each
    """

    # key: user_id, val: item_id list
    user_pos_items = {}

    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()

        if user not in user_pos_items:
            user_pos_items[user] = []

        user_pos_items[user].append(item)

    return user_pos_items

# computes recall@K and precision@K
def RecallPrecision_ATk(groundTruth, r, k):
    """Computers recall @ k and precision @ k

    Args:
        groundTruth (list[list[long]]): list of lists of item_ids. Cntaining highly rated items of each user.
                            In other words, this is the list of true_relevant_items for each user

        r (list[list[boolean]]): list of lists indicating whether each top k item recommended to each user
                            is a top k ground truth (true relevant) item or not

        k (int): determines the top k items to compute precision and recall on

    Returns:
        tuple: recall @ k, precision @ k
    """

    # number of correctly predicted items per user
    # -1 here means I want to sum at the inner most dimension
    num_correct_pred = torch.sum(r, dim=-1)

    # number of items liked by each user in the test set
    user_num_liked = torch.Tensor([len(groundTruth[i]) for i in range(len(groundTruth))])

    recall = torch.mean(num_correct_pred / user_num_liked)
    precision = torch.mean(num_correct_pred) / k
    return recall.item(), precision.item()

# computes NDCG@K
def NDCGatK_r(groundTruth, r, k):
    """Computes Normalized Discounted Cumulative Gain (NDCG) @ k

    Args:
        groundTruth (list): list of lists containing highly rated items of each user
        r (list): list of lists indicating whether each top k item recommended to each user
            is a top k ground truth item or not
        k (int): determines the top k items to compute ndcg on

    Returns:
        float: ndcg @ k
    """
    assert len(r) == len(groundTruth)

    test_matrix = torch.zeros((len(r), k))

    for i, items in enumerate(groundTruth):
        length = min(len(items), k)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = torch.sum(max_r * 1. / torch.log2(torch.arange(2, k + 2)), axis=1)
    dcg = r * (1. / torch.log2(torch.arange(2, k + 2)))
    dcg = torch.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0.
    return torch.mean(ndcg).item()

# wrapper function to get evaluation metrics
def get_metrics(model,
                input_edge_index, # adj_mat based edge index
                input_exclude_edge_indices, # adj_mat based exclude edge index
                k):
    """Computes the evaluation metrics: recall, precision, and ndcg @ k

    Args:
        model (LighGCN): lightgcn model

        edge_index (torch.Tensor): 2 by N list of edges for split to evaluate

        exclude_edge_indices ([type]): 2 by N list of edges for split to discount from evaluation

        k (int): determines the top k items to compute metrics on

    Returns:
        tuple: recall @ k, precision @ k, ndcg @ k
    """
    # get the embedding tensor at layer 0 after training
    user_embedding = model.users_emb.weight
    item_embedding = model.items_emb.weight


    # convert adj_mat based edge index to r_mat based edge index so we have have
    # the first list being user_ids and second list being item_ids for the edge index
    edge_index = convert_adj_mat_edge_index_to_r_mat_edge_index(input_edge_index)

    # This is to exclude the edges we have seen before in our predicted interaction matrix (r_mat_rating)
    # E.g: for validation set, we want want to exclude all the edges in training set
    exclude_edge_indices = [convert_adj_mat_edge_index_to_r_mat_edge_index(exclude_edge_index) \
                                      for exclude_edge_index in input_exclude_edge_indices]



    # Generate predicted interaction matrix (r_mat_rating)
    # (num_users x 64) dot_product (num_item x 64).T
    r_mat_rating = torch.matmul(user_embedding, item_embedding.T)

    # shape: num_users x num_item
    rating = r_mat_rating

    for exclude_edge_index in exclude_edge_indices:
        # gets all the positive items for each user from the edge index
        # it's a dict: user -> positive item list
        user_pos_items = get_user_positive_items(exclude_edge_index)

        # get coordinates of all edges to exclude
        exclude_users = []
        exclude_items = []
        for user, items in user_pos_items.items():
            # [user] * len(item) can give us [user1, user1, user1...] with len of len(item)
            # this makes easier to do the masking below
            exclude_users.extend([user] * len(items))
            exclude_items.extend(items)

        # set the excluded entry in the rat_mat_rating matrix to a very small number
        rating[exclude_users, exclude_items] = -(1 << 10)

    # get the top k recommended items for each user
    _, top_K_items = torch.topk(rating, k=k)

    # get all unique users in evaluated split
    users = edge_index[0].unique()

    # dict of user -> pos_item list
    test_user_pos_items = get_user_positive_items(edge_index)

    # convert test user pos items dictionary into a list of lists
    test_user_pos_items_list = [test_user_pos_items[user.item()] for user in users]


    # r here is "pred_relevant_items ∩ actually_relevant_items" list for each user
    r = []
    for user in users:
        user_true_relevant_item = test_user_pos_items[user.item()]
        # list of Booleans to store whether or not a given item in the top_K_items for a given user
        # is also present in user_true_relevant_item.
        # this is later on used to compute n_rel_and_rec_k
        label = list(map(lambda x: x in user_true_relevant_item, top_K_items[user]))
        r.append(label)

    r = torch.Tensor(np.array(r).astype('float'))

    recall, precision = RecallPrecision_ATk(test_user_pos_items_list, r, k)
    ndcg = NDCGatK_r(test_user_pos_items_list, r, k)

    return recall, precision, ndcg

# wrapper function to evaluate model
def evaluation(model,
               edge_index, # adj_mat based edge index
               exclude_edge_indices,  # adj_mat based exclude edge index
               k,
               lambda_val
              ):
    """Evaluates model loss and metrics including recall, precision, ndcg @ k

    Args:
        model (LighGCN): lightgcn model
        edge_index (torch.Tensor): 2 by N list of edges for split to evaluate
        sparse_edge_index (sparseTensor): sparse adjacency matrix for split to evaluate
        exclude_edge_indices ([type]): 2 by N list of edges for split to discount from evaluation
        k (int): determines the top k items to compute metrics on
        lambda_val (float): determines lambda for bpr loss

    Returns:
        tuple: bpr loss, recall @ k, precision @ k, ndcg @ k
    """
    # get embeddings
    # users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(edge_index)

    # r_mat_edge_index = convert_adj_mat_edge_index_to_r_mat_edge_index(edge_index)

    # edges = structured_negative_sampling(r_mat_edge_index, contains_neg_self_loops=False)

    # user_indices, pos_item_indices, neg_item_indices = edges[0], edges[1], edges[2]

    # users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]

    # pos_items_emb_final, pos_items_emb_0 = items_emb_final[pos_item_indices], items_emb_0[pos_item_indices]

    # neg_items_emb_final, neg_items_emb_0 = items_emb_final[neg_item_indices], items_emb_0[neg_item_indices]


    recall, precision, ndcg = get_metrics(model,
                                          edge_index,
                                          exclude_edge_indices,
                                          k)

    return recall, precision, ndcg

if __name__ == "__main__":

    model = LightGCN(num_users=num_users,
                 num_items=num_movies,
                 K=layers).to(DEVICE)
    model = torch.load('../models/final-model.pt')

    val_edge_index = torch.load('data/val_edge_index.pt')

    K = 20
    LAMBDA = 1e-6
    recall, precision, ndcg = evaluation(model, val_edge_index, [], K, LAMBDA)

    print(f"val_recall@{K}: {round(recall, 5)}, val_precision@{K}: {round(precision, 5)}, val_ndcg@{K}: {round(ndcg, 5)}")