selected_genes = "tcga_prostate_expressed_genes_and_cancer_genes.csv"
data_base = {'data_type': ['mut_important', 'cnv_del', 'cnv_amp'],
             'drop_AR': False,
             'cnv_levels': 3,
             'mut_binary': True,
             'balanced_data': False,
             'combine_type': 'union',  # intersection
             'use_coding_genes_only': True,
             'selected_genes': selected_genes}
