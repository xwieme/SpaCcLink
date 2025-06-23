import time
from tqdm import tqdm
import numpy as np
import ot
import pandas as pd
from scipy.spatial.distance import cdist
from concurrent.futures import ThreadPoolExecutor
from functools import partial


def computeForLR(interaction, lr_counts, dist_matrix, reg=0.01, iternum=200):
    ligand = interaction[0]
    receptor = interaction[1]
    ligand_count = lr_counts.loc[:, ligand]
    receptor_count = lr_counts.loc[:, receptor]
    ligand_distribution = ligand_count.loc[ligand_count>0]
    receptor_distribution = receptor_count.loc[receptor_count>0]
    
    cost_matrix = dist_matrix[ligand_count>0,:][:,receptor_count>0]

   
    w_dist = ot.sinkhorn2(ligand_distribution.values, receptor_distribution.values, cost_matrix/cost_matrix.max(), reg = reg, numItermax=iternum)
    
    return w_dist

def computeForInteractions(lr_counts, lr_db, dist_matrix, max_workers=8, shuffle=False):
    if shuffle:
        lr_counts = lr_counts.sample(frac=1)
    executor = ThreadPoolExecutor(max_workers=max_workers)
    func = partial(computeForLR, lr_counts= lr_counts, dist_matrix=dist_matrix)
    result_iter = executor.map(func, lr_db.values.tolist())
    w_dist = np.round([i for i in result_iter], 5)
    return w_dist

def getLRdistance(adata, lr_db, shuffle_num=200, max_workers=8):
    X= adata.to_df()
    coords = adata.obs[["X","Y"]].values
    X= adata.to_df()
    l_genes = np.unique(lr_db.iloc[:, 0].values)
    r_genes = np.unique(lr_db.iloc[:, 1].values)

    lr_genes = np.unique(np.append(l_genes, r_genes))
    lr_counts = X.loc[:, lr_genes]
    # 计算欧式距离矩阵
    dist_matrix = cdist(coords, coords, metric='euclidean')

    shuffle_list=[]

    print("w_dist")
    t1 = time.time()
    w_dist = computeForInteractions(lr_counts, lr_db, dist_matrix, max_workers = max_workers)
    t2 = time.time()

    print(t2 - t1)

    print("shuffle list")
    shuffle_list = [computeForInteractions(lr_counts, lr_db, dist_matrix, shuffle=True,max_workers = max_workers) for i in tqdm(range(shuffle_num))]
    shuffle_list = np.stack(shuffle_list)
    w_shuffle = np.mean(shuffle_list, axis=0)
    d_ratio = w_dist/(w_shuffle+1e-5)

    p_value = np.sum(shuffle_list<w_dist, axis=0)/shuffle_num

    p_value[d_ratio>1] = np.sum(shuffle_list[:, d_ratio>1]>w_dist[d_ratio>1], axis=0)/shuffle_num
    # np.sum(shuffle_list[:, d_ratio>1]>w_dist, axis=0)/shuffle_num
    result = lr_db.copy().iloc[:, :2]
    result["w_dist"] = w_dist
    result["w_shuffle"] = w_shuffle
    result["d_ratio"] = d_ratio
    result["p_value"] = p_value
    return result

def splitLR(adata, lr_db, top_ratio=0.1, thre_pVal = 0.05, shuffle_num=200, max_workers=8):
    lr_split_result = getLRdistance(adata, lr_db, shuffle_num, max_workers)
    lr_split_result["interactions"] = lr_split_result["ligand"]+"_"+lr_split_result["receptor"]
    thre_index = int(lr_split_result.shape[0]*top_ratio)
    
    result = lr_split_result.sort_values(by=['d_ratio','p_value'])

    short_lr = result.iloc[:thre_index,:]

    short_lr = short_lr.loc[(short_lr.p_value <= thre_pVal) & (short_lr.d_ratio<1),:]

    result = result.sort_values(by=['d_ratio','p_value'],ascending=[False,True])
    # 
    long_lr = result.iloc[:thre_index,:]
    long_lr = long_lr.loc[(long_lr.p_value <= thre_pVal) & (long_lr.d_ratio>1),:]

    lrs = result[["ligand", "receptor","interactions"]]

    lrs['type'] = 'medium-range'
    lrs.loc[lrs['interactions'].isin(short_lr['interactions']), 'type']='short-range'
    lrs.loc[lrs['interactions'].isin(long_lr['interactions']), 'type']='long-range'
    lrs.index = lrs['interactions'].values
    lrs = lrs.sort_values(by='type')
    return lrs
