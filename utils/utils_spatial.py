import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import squareform, pdist
from scipy.sparse import issparse

def cal_RBF(X, l):
    """Convert Euclidean distance to RBF distance"""

    if issparse(X):
        rbf_d = X
        rbf_d[X.nonzero()] = np.exp(-X[X.nonzero()].A**2 / (2 * l ** 2))
        rbf_d.setdiag(1)
    else:
        rbf_d = np.exp(- X**2 / (2 * l ** 2))
        np.fill_diagonal(rbf_d, 1)

    return rbf_d

def getKNN(coords, n_neighbors=32):
    
    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm='ball_tree', 
        metric='euclidean'
    ).fit(coords)
    distances = nbrs.kneighbors_graph(coords, mode='distance')
    return distances

def select_w(coords):
    dist_mat = squareform(pdist(coords, metric='euclidean'))
    dist_mat = dist_mat[dist_mat>0]
    disp = (np.max(dist_mat)-np.min(dist_mat))/np.mean(dist_mat)
    w = np.min(dist_mat)*disp
    return w

def weight_mat(adata):
    coords = adata.obsm['spatial']
    w = select_w(coords)

    n_neighbors = 6
    short_nbr = getKNN(coords, n_neighbors)
    short_rbf = cal_RBF(short_nbr, w)

    medium_nbr = getKNN(coords, 4*n_neighbors)
    medium_rbf = cal_RBF(medium_nbr, 2*w)

    long_nbr = getKNN(coords, 16*n_neighbors)
    long_rbf = cal_RBF(long_nbr, 4*w)
    
    adata.obsm['short_weight'] = short_rbf
    adata.obsm['medium_weight'] = medium_rbf
    adata.obsm['long_weight'] = long_rbf
    return 
    

def z_normalize(data, axis=0):
    mean = np.mean(data, axis) 
    N = data.shape[0]
    X = data - mean
    normalized_data = X / np.sqrt(np.sum(X**2, axis=axis)/N)
    return normalized_data

def spatial_global_I(X, groups, short_rbf, medium_rbf, long_rbf, shuffle=False):
    if shuffle:
        X = X.sample(frac=1)
    bmi = []
    for lr_type, group in groups:
        if lr_type=='short-range':
            rbf = short_rbf
        elif lr_type=="medium-range" :
            rbf = medium_rbf
        else:
            rbf = long_rbf
       
        l_data = X[group['ligand']].values
        r_data = X[group['receptor']].values

        bmi.append(
            pd.DataFrame(np.sum(np.dot(rbf.toarray(), l_data)*r_data, axis=0),
                          index = group.index, columns=['Global_I'])
            )

    bmi = pd.concat(bmi)
    return bmi



def spatial_statistics_global(X, lrs, short_rbf, medium_rbf, long_rbf, iter_num=1000):
    
    groups = lrs.groupby("type")
    short_rbf = short_rbf * X.shape[0]/short_rbf.sum()
    medium_rbf = medium_rbf * X.shape[0]/medium_rbf.sum()
    long_rbf = long_rbf * X.shape[0]/long_rbf.sum()
    bmi = spatial_global_I(X, groups, short_rbf, medium_rbf, long_rbf)
    perm_bmi = [spatial_global_I(X, groups, short_rbf, medium_rbf, long_rbf, shuffle=True) 
            for i in range(iter_num)]
    perm_bmi = pd.concat(perm_bmi,axis=1).values

    p = np.sum(np.sign(bmi.values)*(perm_bmi)>=np.sign(bmi.values)*bmi.values, axis=1)/iter_num

    
    bmi['p_value'] = p
    return bmi

def spatial_statistics_local(X, lrs, short_rbf, medium_rbf, long_rbf, iter_num=1000):
    
    groups = lrs.groupby("type")
    short_rbf = short_rbf * X.shape[0]/short_rbf.sum()
    medium_rbf = medium_rbf * X.shape[0]/medium_rbf.sum()
    long_rbf = long_rbf * X.shape[0]/long_rbf.sum()
    
    bmi = spatial_local_I(X, groups, short_rbf, medium_rbf, long_rbf)
    
    batch=50
    p = 0
    for i in range(int(iter_num/batch)):
        perm_bmi = [spatial_local_I(X, groups, short_rbf, medium_rbf, long_rbf, shuffle=True).values
                for i in range(batch)]
        perm_bmi = np.stack(perm_bmi, axis=0)
        p = (p+np.sum(np.sign(bmi.values)*perm_bmi>=np.sign(bmi.values)*bmi.values,axis=0))
    p = p/iter_num
    p[np.bitwise_and(X[lrs['ligand']].values<=0, X[lrs['receptor']].values<=0)] = 1
    p = pd.DataFrame(p, index=bmi.index, columns=bmi.columns)
    
    return bmi,p

def spatial_local_I(X, groups, short_rbf, medium_rbf, long_rbf, shuffle=False):
    if shuffle:
        X = X.sample(frac=1)

    bmi = []
    for lr_type, group in groups:
        if lr_type=='short-range':
            rbf = short_rbf
        elif lr_type=="medium-range" :
            rbf = medium_rbf
        else:
            rbf = long_rbf
            
        l_data = X[group['ligand']].values
        r_data = X[group['receptor']].values
        local_i_t = np.dot(rbf.toarray(), l_data)*r_data+np.dot(rbf.toarray(), r_data)*l_data
        local_i_t = local_i_t
        bmi.append(
            pd.DataFrame(local_i_t, columns=group['ligand']+"_"+group['receptor'], dtype=np.float16)
        )

    bmi = pd.concat(bmi,axis=1)
    bmi = bmi.set_index(X.index)
    
    return bmi

def spatial_statistics(adata, lrs, iter_num=500):
    weight_mat(adata)
    X = z_normalize(adata.to_df())
    adata.obsm['norm_X'] = X
    short_rbf = adata.obsm['short_weight']
    medium_rbf = adata.obsm['medium_weight']
    long_rbf = adata.obsm['long_weight']
    tmp = spatial_statistics_global(X, lrs, short_rbf, medium_rbf, long_rbf, iter_num=iter_num)  

    result_global = pd.merge(lrs, tmp, left_index=True, right_index=True, how='inner')
    pairs = result_global.loc[result_global['p_value']<=0.05]
    local_i, p = spatial_statistics_local(X, pairs, short_rbf, medium_rbf, long_rbf, iter_num=iter_num)
    local_i.iloc[p.values>0.05] = 0
    select_id = np.sum(p.values<=0.05,axis=0)>=3
    local_i = local_i.loc[:, select_id]
    pairs = pairs.loc[local_i.columns]
    return pairs, local_i