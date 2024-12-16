import os
import random
import numpy as np
import igraph as ig
import networkx as nx
import seaborn as sns
from igraph import Graph
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
def getcolors(n):
    
    # Colors
    colors = ["#51c4c2","#4583b3","#f78e26","#f172ad","#f7afb9","#be86ba","#8b66b8",
              "#70c17f","#FFE34F",'#f23b27', '#d062fb', '#80A6E2',"#0094ff",
              "#fabb6e", "#afd8ad", "#0075c5","#8481BA","#cedfef"]
    if n <= len(colors):
        selected_colors = random.sample(colors, n)
    else:
        # 获取所有可用颜色的名称列表
        colors = list(mcolors.CSS4_COLORS)
        
        # 从颜色列表中随机选择n个不同的颜色
        selected_colors = random.sample(colors, n)
    return selected_colors


def adjust_color(color, alpha=0.6):
    rgba_color = mcolors.to_rgba(color)  # 将颜色转换为 RGBA 格式

    adjusted_color = (
        rgba_color[0],
        rgba_color[1],
        rgba_color[2],
        alpha
    )

    return adjusted_color


def radian_rescale(x, start=0, direction=1):
    def c_rotate(x):
        return (x + start) % (2 * np.pi) * direction
    
    return c_rotate(np.interp(x, (min(x), max(x)), (0, 2 * np.pi)))

def calculate_angle(coords_scale, g):
    coords_scale = np.array(coords_scale)
    angles = []
    for node in g.vs:
        if coords_scale[node.index, 0] > 0:
            angle = -np.arctan(coords_scale[node.index, 1] / coords_scale[node.index, 0])
        else:
            angle = np.pi - np.arctan(coords_scale[node.index, 1] / coords_scale[node.index, 0])
        angles.append(angle)
    return angles

def pl_heatmap(df, title, xlabel, ylabel, barlabel, cmap=None, save_path=None, figsize = (10,5)):
    if cmap == None:
        colors = ['#ffffff','#f6b293', '#d72e25',"#c90220","#470101"]
        cmap = LinearSegmentedColormap.from_list('colors',colors,N=256)
    
    
    plt.figure(figsize = figsize)

    # a = sns.cubehelix_palette(rot=-0.05,start=-2, gamma=1.5, hue = 2.5, dark = 0.6, light=1)
    ax = sns.heatmap(df, cmap=cmap, annot=False,square=True, linewidths=.9,
                     cbar_kws={"shrink": .5, "pad": 0.1, "label":barlabel,},
                     xticklabels=True, yticklabels=True)


    # 获取当前横轴和纵轴刻度标签
    xtick_labels = ax.get_xticklabels()
    ytick_labels = ax.get_yticklabels()
    
    # 设置横轴和纵轴刻度标签的字体属性
    ax.set_xticklabels(xtick_labels, fontsize=14, fontweight='light', fontfamily='Times New Roman')
    ax.set_yticklabels(ytick_labels, fontsize=14, fontweight='light', fontfamily='Times New Roman')
    # 设置横轴和纵轴刻度标签的方向
    ax.tick_params(axis='x', labelrotation=90)
    ax.tick_params(axis='y', labelrotation=0)

    cax = plt.gcf().axes[-1]
    
    cax.set_ylabel(cax.get_ylabel(), fontsize=16, fontweight='bold', fontfamily='Times New Roman',labelpad=10)
    cax.tick_params(labelsize=12)
    
    plt.xlabel(xlabel,fontsize=14, labelpad=15)
    plt.ylabel(ylabel,fontsize=14, labelpad=15)
    plt.title(title, fontsize=14, pad=20, fontweight='bold', loc='center')
    
    if save_path != None:
        plt.savefig(save_path, bbox_inches = 'tight')
    plt.show()
    
def pl_circle(df, colors=None, min_percentile=10, save_path="circle.pdf"):

    edge_weight = df.values.copy()
    labels = df.index.tolist()

    vertex_size = 15
    # vertex_size = vertex_weight/np.max(vertex_weight)*15+5
    vertex_label_size= 12
    edge_width_max = 2
    vertex_label_dist = 3
    
    min_thre = np.percentile(edge_weight[edge_weight>0], q=min_percentile)
    edge_weight[edge_weight <= min_thre] = 0
    edge_width = edge_weight/np.max(edge_weight)*edge_width_max
    edge_width[edge_width>0] += 0.1
    
    if colors == None:
        colors = getcolors(len(labels))
    else:
        colors = [colors[key] for key in labels]

    g = Graph.Weighted_Adjacency(edge_width)

    # 创建一个图对象
    g.vs["color"] = colors
    g.es["width"] = g.es["weight"]
    # 设置边颜色为源节点的颜色
    for edge in g.get_edgelist():
        source_color = g.vs[edge[0]]["color"]
        g.es[g.get_eid(edge[0], edge[1])]["color"] = adjust_color(source_color)


    margin=80
    edge_arrow_size = 0.5
    edge_arrow_width = 0.8
    edge_curved = 0.1
    # 绘制circle布局图
    layout = g.layout("circle")
    layout = (np.array(layout)-np.mean(layout))/np.std(layout)
    
    g.vs["label_angle"] = radian_rescale(x=np.arange(1, len(g.vs) + 2), direction=-1, start=0)
    



    visual_style = dict(vertex_size = vertex_size,vertex_label_color="black", 
                        vertex_label_size=vertex_label_size, 
                        vertex_label=labels, vertex_label_dist=vertex_label_dist,
                        edge_curved = edge_curved, edge_arrow_size =edge_arrow_size,
                        edge_arrow_width = edge_arrow_width,
                         margin=margin, layout=layout
                        )
    
    
    ig.plot(g, target=save_path,  **visual_style, bbox=(10, 10,300,300))

def pl_pathway(pathways, rec_tftg, celltype, receptor, save_network_dir = None):
    path, tfs, tgs = pathways[celltype][receptor]
    tftg_path = rec_tftg.loc[(rec_tftg.receptor==receptor)& (rec_tftg.tf.isin(tfs))&(rec_tftg.dest.isin(tgs))]
    tftg_path  = list(zip(tftg_path.tf, tftg_path.dest))
    path += tftg_path
    path = tuple(set(path))
    G = nx.Graph() 
    G.add_edges_from(path)
    plt.figure(figsize=(20, 10))

    node_types = dict()
    for node in G:
        if node in receptor:
            node_types[node] = "Receptor"
        elif node in tgs:
            node_types[node] = "Target"
        elif node in tfs:
            node_types[node] = "Tf"
        else:
            node_types[node] = "Mediator"
    nx.set_node_attributes(G, node_types, "type")
    node_colors = {'Receptor':'#70c17f',"Mediator":'#cedfef', "Tf": '#f172ad', "Target":'#0094ff'}
    pos=nx.kamada_kawai_layout(G) 

    nx.draw_networkx(G, pos=pos, with_labels=True, node_color=[node_colors[G.nodes[node]["type"]] for node in G.nodes()], font_size=11, font_weight="bold",node_size=800, width=1.5, alpha=0.8)

    if save_network_dir!=None:
        nx.write_graphml_lxml(G, os.path.join(save_network_dir,"downstream.graphml"))
    # 创建图例
    legend_handles = [plt.Line2D([], [], marker='o', color=color, linestyle='None', markersize=10) for color in node_colors.values()]
    plt.legend(legend_handles, node_colors.keys(),fontsize=13)

    plt.axis('off')
    plt.show()
        
    
            