from network.reactome import ReactomeNetwork
import itertools
import numpy as np 
import pandas as pd 
import logging

gtex_file = "_database/GTEx_tissue_specific_gene_expression_binary.csv"
hugo_file = '_database/HUGO_genes/protein-coding_gene_with_coordinate_minimal.txt'
def get_tissue_specific_maps(genes, Network_params, use_coding_genes_only=True):
    tissue_maps = {}
    genes = set(genes)
    gtex = pd.read_csv(gtex_file,index_col="Description")
    genes.intersection_update(gtex.index)
    df = gtex.loc[list(genes),Network_params.tissues]
    if use_coding_genes_only:
        coding_genes_df = pd.read_csv(hugo_file, sep='\t', header=None)
        coding_genes_df.columns = ['chr', 'start', 'end', 'name']
        coding_genes = set(coding_genes_df['name'].unique())
        coding_genes.intersection_update(df.index)
        df = df.loc[list(coding_genes),:]
    print("Genes number in tissue:")
    print(df.sum())
    print('#'*50)
    for t in Network_params.tissues:
        print(f'{t}')
        maps = get_layer_maps(genes = df[df.loc[:,t]].index,
                              n_levels = Network_params.n_hidden_layers,
                              direction = Network_params.direction,
                              add_unk_genes=Network_params.add_unk_genes)
        tissue_maps[t] = maps
        print('#'*50)
    return tissue_maps

def get_map_from_layer(layer_dict):
    pathways = list(layer_dict.keys())
    print('pathways', len(pathways))
    genes = list(itertools.chain.from_iterable(list(layer_dict.values())))
    genes = list(np.unique(genes))
    print('genes', len(genes))

    n_pathways = len(pathways)
    n_genes = len(genes)

    mat = np.zeros((n_pathways, n_genes))
    for p, gs in list(layer_dict.items()):
        g_inds = [genes.index(g) for g in gs]
        p_ind = pathways.index(p)
        mat[p_ind, g_inds] = 1

    df = pd.DataFrame(mat, index=pathways, columns=genes)
    # for k, v in layer_dict.items():
    #     print k, v
    #     df.loc[k,v] = 1
    # df= df.fillna(0)
    return df.T


def get_layer_maps(genes, n_levels, direction, add_unk_genes):
    reactome_layers = ReactomeNetwork().get_layers(n_levels, direction)
    filtering_index = genes
    maps = []
    for i, layer in enumerate(reactome_layers[::-1]):
        print('layer #', i)
        mapp = get_map_from_layer(layer)
        filter_df = pd.DataFrame(index=filtering_index)
        print('filtered_map', filter_df.shape)
        filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='left')
        # filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='inner')
        print('filtered_map', filter_df.shape)
        # filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='inner')

        # UNK, add a node for genes without known reactome annotation
        if add_unk_genes:
            print('UNK ')
            filtered_map['UNK'] = 0
            ind = filtered_map.sum(axis=1) == 0
            filtered_map.loc[ind, 'UNK'] = 1
        ####

        filtered_map = filtered_map.fillna(0)
        print('filtered_map', filter_df.shape)
        # filtering_index = list(filtered_map.columns)
        filtering_index = filtered_map.columns
        logging.info('layer {} , # of edges  {}'.format(i, filtered_map.sum().sum()))
        maps.append(filtered_map)
    return maps