import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.utils import getTopCorrPairs,calCorr

def build_dataset(adata, pathways, save_dir, cut_off=0.1, thre_cor = None, test_percent=0.2, neg_sampling=1.0):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
    
    genes_interest = np.unique(pathways.iloc[:, :2].values)

    data = adata.to_df()
    data = data[genes_interest]
    corr = calCorr(data)

    maps = dict(zip(genes_interest, range(len(genes_interest))))
    links = pathways.iloc[:, :2].values
    links =  np.vectorize(maps.get)(links)
    corr_values = corr.values
    corr_values = corr_values[links[:,0], links[:,1]]
    pathways["weight"] = corr_values
    topCorrPairs = getTopCorrPairs(corr, cutoff=cut_off, thre_cor=thre_cor)
    top_pathways = pd.merge(pathways, topCorrPairs, left_on=pathways.columns.values.tolist()[:2], right_on=topCorrPairs.columns.values.tolist(), how="inner")
    utils_build(pathways, top_pathways, adata.var_names, save_dir, test_percent, neg_sampling)
    adata.to_df().to_csv(os.path.join(save_dir, "express_matrix.csv"))
    pathways.to_csv(os.path.join(save_dir, "pathways.csv"))
    return 

def utils_build(db, toppairs, genes, save_dir, test_percent=0.2, neg_sampling=1.0):
    

    groups = toppairs.groupby(toppairs.columns[0])
    
    data = pd.DataFrame(columns=["src", "dst_pos", "dst_neg"])
    for src_gene, group in groups:

        pos_dst = group.iloc[:, 1].values
        n_pos = group.shape[0]
        
        n_neg_sampling = int(neg_sampling * n_pos)
        
        neg_dst = np.setdiff1d(genes, db.loc[db.iloc[:, 0] == src_gene, db.columns[1]].unique())
        neg_dst = np.random.choice(neg_dst, size=n_neg_sampling)
        

        new_rows = pd.DataFrame({"src": np.repeat(src_gene, n_neg_sampling),
                                 "dst_pos": np.repeat(pos_dst, neg_sampling),
                                  "dst_neg": neg_dst
                                 })

        data = pd.concat([data, new_rows])

    train_data, test_data = train_test_split(data, test_size=test_percent, shuffle=True)

    train_data.to_csv(os.path.join(save_dir, "train_data.csv"), index=False)
    test_data.to_csv(os.path.join(save_dir, "test_data.csv"), index=False)

    print(os.path.join(save_dir, "train_data.csv"))
    print(train_data.shape)
    print(test_data.shape)
