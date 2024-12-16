import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx


def switchId(data, mapping):
    
    for i in data.columns:
        data[i] = data[i].map(mapping)
        
    data = torch.LongTensor(data.values).contiguous()
    
    return data



def getLinks(links, mapping):
    G = nx.from_pandas_edgelist(links, source=links.columns[0], target=links.columns[1])
    G = nx.relabel_nodes(G, mapping)
    edges_index,_ = from_networkx(G)
    edges = edges_index[1].view(1, -1)
    nodes = torch.LongTensor(list(G.nodes()))
    edges = nodes[edges].view(2, -1)
    return edges

def getData(express_matrix, pathways, mapping):
    links = getLinks(pathways, mapping)
    x = torch.FloatTensor(express_matrix.T.to_numpy())
    edges = SparseTensor(row=links[0], col=links[1], sparse_sizes=(x.shape[0], x.shape[0]))
    data = Data(x, edges=edges)
    return data

def calCorr(data):
    # corr = data.corr(method=method)
    X = data.values

    X = (X-np.mean(X,axis=0))/np.std(X,axis=0)

    r = np.dot(X.T,X)/X.shape[0]
    r = pd.DataFrame(r, columns = data.columns, index = data.columns)
    return r

def getTopCorrPairs(corr, cutoff=None, thre_cor = 0.1):

    if cutoff==None:
        thre_cor = thre_cor
    else:
        values = corr.abs().values
        values = values[values > 0]
        values = np.sort(values)[::-1]
        thre_cor = values[int(len(values) * cutoff)]
    row, col = np.where(corr.abs().values > thre_cor)
    
    nodes = corr.columns.values
    topCorrPairs = pd.DataFrame({"src": nodes[row], "dest": nodes[col]})
    return topCorrPairs