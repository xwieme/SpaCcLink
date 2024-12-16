import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import coo_matrix
from scipy.stats import fisher_exact
import networkx as nx

def spatial_cell2cell(X, source_type, target_type, labels, groups, short_rbf, medium_rbf, long_rbf, shuffle=False):
    if shuffle:
        X = X.sample(frac=1)

    cci_matrix = np.zeros((len(source_type)* len(target_type), np.sum(groups.size())))

    for lr_type, group in groups:
        
        if lr_type=='short-range':
            rbf = short_rbf
        elif lr_type=="medium-range" :
            rbf = medium_rbf
        else:
            rbf = long_rbf
            
        l_data = X[group['ligand']]
        r_data = X[group['receptor']]
        for i in range(len(source_type)):
            gi = source_type[i]
            cells_in_gi = (labels==gi)
            t_l_data = l_data.loc[cells_in_gi, :].values
            cond1 = np.mean(t_l_data, axis=0)<=0
            for j in range(len(target_type)):
                gj = target_type[j]
                cells_in_gj = (labels==gj)
                t_r_data = r_data.loc[cells_in_gj, :].values

                t_rbf = rbf.A[cells_in_gj,:][:, cells_in_gi]
                tmp = np.sum(t_rbf)
                if tmp == 0:
                    break
                scale = (t_rbf.shape[0]+t_rbf.shape[1])/(2*tmp)
                t_rbf = t_rbf*scale
                tmp = np.sum(np.dot(t_rbf, t_l_data)*t_r_data, axis=0)
                cond2 = np.mean(t_r_data, axis=0)<=0
                tmp[np.bitwise_or(cond1, cond2)] = -1
                
                cci_matrix[i*len(target_type)+j, group.index.tolist()] = tmp
    return cci_matrix


def getCelltypeInteraction(adata, lrs, source_types, target_types, groupby, iter_num=500,thre_p=0.05):
    X = adata.obsm['norm_X']
    short_rbf = adata.obsm['short_weight']
    medium_rbf = adata.obsm['medium_weight']
    long_rbf = adata.obsm['long_weight']
    clusters = adata.obs[groupby]
    grouped = lrs.reset_index().groupby("type")
    celltype_lr_interaction = spatial_cell2cell(X, source_types, target_types, clusters, grouped, short_rbf, medium_rbf, long_rbf)
    p_celltype_interaction = [spatial_cell2cell(X, source_types, target_types, clusters, grouped, short_rbf, medium_rbf, long_rbf, shuffle=True)
            for i in range(iter_num)]
    p_celltype_interaction = np.stack(p_celltype_interaction, axis=0)
    p_celltype_interaction = np.sum(p_celltype_interaction>=celltype_lr_interaction,axis=0)/iter_num
    
    cellpairs = [s +"--"+t for s in source_types for t in target_types]
    celltype_lr_interaction = pd.DataFrame(celltype_lr_interaction, index=cellpairs, columns=lrs['interactions'])
    p_celltype_interaction = pd.DataFrame(p_celltype_interaction, index=cellpairs, columns=lrs['interactions'])
    
    ix, iy = np.where(p_celltype_interaction<=thre_p)
    celltype_interaction = pd.DataFrame({
        "interaction": np.array(cellpairs)[ix],
        "lr": lrs['interactions'].values[iy],
        "moranI": celltype_lr_interaction.values[p_celltype_interaction<=thre_p].round(3),
        "p_val": p_celltype_interaction.values[p_celltype_interaction<=thre_p] 
    })
    celltype_interaction['Flog10P'] = -np.log10(celltype_interaction['p_val']+1e-10).round(3)
    celltype_interaction = celltype_interaction.loc[celltype_interaction["moranI"]>0]
    
    return celltype_interaction


def cal_Influence_score(adata, relations, groupby='celltype'):
    clusters = adata.obs[groupby]
    groups = np.unique(clusters)

    x = adata.to_df()
    means = np.zeros((groups.shape[0], x.shape[1]))

    for i, g in enumerate(groups):
        
        g_x = x[clusters==g]
        means[i] = np.mean(g_x, axis=0)
        
    sums = np.sum(means, axis=1)
    total_sum = np.sum(sums)

    genes_row_sum = np.sum(means, axis=0)

    # 基因i在细胞类型A中表达的零分布
    celltype_expected = np.outer(sums/total_sum, genes_row_sum)
    mapping = dict(zip(groups, range(groups.shape[0])))

    clusters_ = np.vectorize(mapping.get)(np.array(clusters))

    temp =  celltype_expected[clusters_, :]

    pem = x/temp
    pem = np.log2(pem+1)
    pem = pd.DataFrame(data=pem, columns=x.columns)
    grouped = relations.groupby("receptor")
    module_score = pd.DataFrame(columns=list(grouped.groups.keys()))
    for i, group in grouped:
        if group.shape[0] < 3: 
            continue
        dest_pem = pem[group['dest']]
        module_score[i] = pem[i].values*np.sum(dest_pem.values *  group['relation'].values, axis=1) / dest_pem.shape[1]  
    return module_score.dropna(axis=1)

def computeCciScore(adata, lr_db, module_score, local_i, thre_percent = 50):

    distances = pdist(adata.obs[["X","Y"]])
    distance = squareform(distances)
    data = adata.to_df()
    cci_matrix = dict()
    k = 0
    for index, row in lr_db.iterrows():
        l = row['ligand']
        r = row['receptor']
        
        idx = np.where(local_i[index]!=0)[0]
        t_dist = distance[idx,:][:, idx]

        d_thre = np.percentile(t_dist, thre_percent)

        in_idx = t_dist<=d_thre
        t_dist[~in_idx] = 0
        # t_dist[in_idx] = np.exp(-t_dist[in_idx]**2/(2*d_thre**2))
        t_dist[in_idx] = 1-t_dist[in_idx]/d_thre
        t_dist[t_dist<1e-5] = 0
        np.fill_diagonal(t_dist, 0)
        l_data = data[l].values[idx]
        r_data = data[r].values[idx]
        r_score = module_score[r].values[idx]
        val = np.outer(l_data, r_data)*r_score*t_dist
        id1, id2 = np.where(val>0)
        cci_matrix[index] = coo_matrix((val[id1,id2], (idx[id1], idx[id2])), shape=(data.shape[0], data.shape[0]))
    return cci_matrix

def computeCelltypeCciScore(cci_matrix, label, lrpair):
    cell_cci = None
    if lrpair == None:      # 累加所有lrpair的score
        for k,v in cci_matrix.items():
            if cell_cci is None:
                cell_cci = v.A
            else:
                cell_cci += v.A
    else:
        cell_cci = cci_matrix[lrpair].A
        
    groups = np.unique(label)
    celltype_cci = np.zeros((groups.shape[0],groups.shape[0]))
    for i in range(groups.shape[0]):
        gi = groups[i]
        cells_in_gi = np.where(label==gi)[0]
        if cells_in_gi.shape[0]<10: continue
        for j in range(i,groups.shape[0]):
            
            gj = groups[j]
            cells_in_gj = np.where(label==gj)[0]
            if cells_in_gj.shape[0] < 10: continue
            a = cell_cci[cells_in_gi ,:][:,cells_in_gj]
            b = cell_cci[cells_in_gj ,:][:,cells_in_gi]
            celltype_cci[i,j] = np.mean(a)
            celltype_cci[j,i] = np.mean(b)
    celltype_cci = pd.DataFrame(celltype_cci, index=groups, columns=groups)
    return celltype_cci


def get_interest_receptor(celltype_interaction):
    receivers = []

    for _,row in celltype_interaction.iterrows():
        receivers.append(row.interaction.split("--")[1])
    celltype_interaction['receiver'] = receivers 
    
    lr_genes = celltype_interaction['lr'].str.split('_', expand=True)
    lr_genes.columns = ['ligand', 'receptor']
    celltype_interaction = pd.concat([celltype_interaction, lr_genes], axis=1)  
    receptors = lr_genes['receptor'].unique()
    groups = celltype_interaction.groupby('receiver')
    activated_receptor = dict()
    for celltype,g in groups:
        activated_receptor[celltype] = g.receptor.values
    return receptors, activated_receptor

def _fisher(adata, activated_receptor, relations, rec_tftg,  groupby='celltype', cut_off=0.1, batchsize=10000):
    N = adata.var_names.shape[0]
    data = adata.to_df()
    clusters = adata.obs[groupby]

    fisher_result = dict()
    ct_rectg = dict()
    for celltype in activated_receptor.keys():
        print(celltype)
        celltype_data = data.loc[clusters==celltype]
        receptors = activated_receptor[celltype]
        
        # Identify the tg associated with receptors and calculate the co-expression relationship in each cell type
        ct_relations = relations.loc[relations.receptor.isin(receptors)] 
        columns = list(relations.columns).append(['weight'])
        res_celltype = pd.DataFrame(columns=columns)
        
        n =int(np.ceil(ct_relations.shape[0]/batchsize))
        for i in range(n):
            tmp = ct_relations.iloc[i*batchsize:(i+1)*batchsize]
            co_ratio = np.sum((celltype_data[tmp.receptor].values)*(celltype_data[tmp.dest].values),axis=0)/celltype_data.shape[0]
            # tmp.loc[:, 'weight'] = co_ratio
            tmp = tmp.assign(weight=co_ratio)
            res_celltype = pd.concat([res_celltype,tmp.iloc[np.where(co_ratio>cut_off)[0]]])
        ct_rectg[celltype] = res_celltype
            
            
        # Identify the tf_tg associated with the receptors
        ct_rec_tftg = rec_tftg.loc[rec_tftg.receptor.isin(receptors)]
        grouped = ct_rec_tftg.groupby(['receptor','tf'])
        
        tmp = pd.DataFrame(grouped.groups.keys(), columns=['receptor', 'tf'])
        pvalue = []

        # Fisher exact
        for (receptor, tf), group in grouped:
            rectg = res_celltype.loc[res_celltype.receptor==receptor]
            a = np.intersect1d(rectg.dest, group.dest).shape[0]
            b = group.shape[0] - a
            c = rectg.shape[0] - a 
            d = N-a-b-c
                
            # 构建2x2列联表
            observed = [[a, b], [c, d]]  # 观察到的频数

            # 执行Fisher精确检验
            _, p_value = fisher_exact(observed)

            pvalue.append(p_value)

        tmp['pval'] = pvalue
        tmp = tmp.loc[tmp.pval<=0.05]
        tmp = tmp.sort_values('pval')
        fisher_result[celltype] = tmp
    return fisher_result, ct_rectg


def findReceptorToTargetPath(adata, celltype_interaction, relations, pathways, rec_tf, tftg, groupby='celltype', cut_off = 0.1):
    
    batchsize = 10000
    n = int(np.ceil(pathways.shape[0]/batchsize))
    res = dict()
    data = adata.to_df()>0
    clusters = adata.obs[groupby]
    groups = np.unique(clusters)
    
    # -----Find the receptors associated with cell-cell communication in each cell type-------------
    receptors, activated_receptor = get_interest_receptor(celltype_interaction)
    
    # -----Fisher_exact-------------        
    rec_tf = rec_tf.loc[rec_tf.receptor.isin(receptors)]
    rec_tftg = rec_tf.merge(tftg,how="inner", left_on='tf', right_on='src')
    rec_tftg = rec_tftg.iloc[:, [0,1,4]]
    rec_tftg = rec_tftg.loc[rec_tftg.receptor != rec_tftg.dest]
    
    fisher_res, ct_rectg = _fisher(adata, activated_receptor, relations, rec_tftg, groupby)
    # -----Calculate the coexpression relationship of genes in pathway-------------        
    ct_activted_pathways = dict()
    path_cor = dict()
    for celltype in groups:
        celltype_data = data.loc[clusters==celltype]
        columns = list(pathways.columns).append(['weight'])
        res_celltype = pd.DataFrame(columns=columns)
        for i in range(n):
            tmp = pathways.iloc[i*batchsize:(i+1)*batchsize]
            co_ratio = np.sum((celltype_data[tmp.src].values)*(celltype_data[tmp.dest].values),axis=0)/celltype_data.shape[0]
            # tmp.loc[:, 'weight'] = co_ratio
            tmp = tmp.assign(weight=co_ratio)
            res_celltype = pd.concat([res_celltype,tmp.iloc[np.where(co_ratio>cut_off)[0]]])
        path_cor[celltype] = res_celltype
    
    # ------------------Find the shortest path---------------------------
    for celltype in activated_receptor.keys():
        ct_pathways = path_cor[celltype]
        ct_pathways['weight'] = ct_pathways['weight'] = 1/ct_pathways.weight
        G = nx.from_pandas_edgelist(ct_pathways, "src","dest","weight")

        largest_comp = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_comp)
        
        ct_fisher_rectf = fisher_res[celltype]
        ct_fisher_rectf = ct_fisher_rectf.loc[ct_fisher_rectf.receptor.isin(G.nodes) & ct_fisher_rectf.tf.isin(G.nodes)]
        fisher_res[celltype] = ct_fisher_rectf
        
        shortest_paths = dict()
        for _, row in ct_fisher_rectf.iterrows():
            start  = row.receptor
            end = row.tf
            path = nx.shortest_path(G, start, end)
            shortest_paths[str(start)+"_"+str(end)] = path
        ct_activted_pathways[celltype] = shortest_paths 
    adata.uns['ct_receptor'] = activated_receptor
    adata.uns['ct_activted_pathways'] = ct_activted_pathways
    adata.uns['fisher_result'] = fisher_res
    adata.uns['ct_receptor_tg'] =  ct_rectg
    adata.uns['rec_tf_tg'] = rec_tftg
    return

def findTopPath(adata, n_target, n_tf):
    activated_receptor = adata.uns['ct_receptor']
    ct_activted_pathways = adata.uns['ct_activted_pathways']
    fisher_result = adata.uns['fisher_result']
    ct_rectg = adata.uns['ct_receptor_tg']
    rec_tftg = adata.uns['rec_tf_tg']
    
    top_activted_pathways = dict()
    for celltype in activated_receptor.keys():
        paths = ct_activted_pathways[celltype]
        ct_fisher_rectf = fisher_result[celltype]
        ct_fisher_rectf.index = ct_fisher_rectf.receptor+"_"+ct_fisher_rectf.tf
        
        rectg = ct_rectg[celltype]
        group = rectg.groupby('receptor')
        top_rectftgs = dict()
        print(celltype)
        for receptor, g in group:
            
            tmp_tgs = g.dest.values[:n_target]
            regulons = rec_tftg.loc[(rec_tftg.receptor==receptor)& (rec_tftg.dest.isin(tmp_tgs))]
            tmp = ct_fisher_rectf.loc[ct_fisher_rectf.receptor==receptor]
            gp = regulons.groupby('dest')
            
            tfs=[]
            tgs=[]
            
            for tg, gg in gp:
                tmp_tfs = tmp.loc[tmp.tf.isin(gg.tf.values)].tf[:n_tf].tolist()
                if len(tmp_tfs) > 0:
                    tgs.append(tg)
                    tfs += tmp_tfs
            tfs = np.unique(tfs)  
            if tfs.shape[0]>0:
                tmp_paths =  []
                for tf in tfs:
                    path = paths[str(receptor)+"_"+str(tf)]
                    tmp_paths += list(zip(path[:-1], path[1:]))
                top_rectftgs[receptor] = [tmp_paths, tfs, tgs]

        top_activted_pathways[celltype] = top_rectftgs
    return top_activted_pathways