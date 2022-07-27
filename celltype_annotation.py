import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from matplotlib.colors import rgb2hex

from pipeline_functions import training_data_select
from pipeline_functions import viz_training_data
from pipeline_functions import one_hot_encode
from pipeline_functions import cell_type_classifier
from pipeline_functions import process_label

sc.set_figure_params(figsize=(4, 4), fontsize=20)
oj = sc.read('/Users/katebridges/Downloads/OJ_24_48_woundbeds.h5ad')

# generating 2D embedding for visualization
sc.tl.pca(oj, svd_solver='auto')
# sc.pl.pca(oj, color='sample')
sc.pp.neighbors(oj)
sc.tl.umap(oj)
sc.pl.umap(oj, color='sample')

# set up for cell type labeling pipeline
marker = ['Cd3d', 'Cd3e', 'Cd3g',
          'Ncr1', 'Klri2',
          'Cxcr2', 'Lcn2', 'Hdc', 'Ly6g', 'S100a8', 'S100a9',
          'Ms4a2', 'Il4',
          'Itgam', 'Arg1', 'Adgre1',
          'Flt3', 'Cd83',
          'Col1a1', 'Dcn', 'Sparc', 'Ptprc',
          ]

cell_types = ['T cell', 'NK cell', 'Neutrophil', 'Basophil', 'Mac/mono', 'DC', 'Fibroblast']

celltype_map = {-1: 'Poorly classified',
                0: 'T cell',
                1: 'NK cell',
                2: 'Neutrophil',
                3: 'Basophil',
                4: 'Macrophage/monocyte',
                5: 'DC',
                6: 'Fibroblast'}

celltypes = np.zeros((len(cell_types), len(marker)))
celltypes[0, :3] = [1, 1, 1]  # T cell
celltypes[0, 13:18] = [-1, -1, -1, -1, -1]
celltypes[1, :5] = [-1, -1, -1, 1, 1]  # NK
celltypes[2, 5:11] = [1, 1, 1, 0, 1, 1]  # neutrophil
celltypes[3, 11:13] = [1, 1]  # basophil
celltypes[4, 13:16] = [1, 1, 1]  # macrophage
celltypes[5, 16:18] = [1, 1]  # DC
celltypes[6, 18:] = [1, 1, 1, -1]  # fibroblast

tot_lab, tot_ideal_ind, tot_traindata, tot_testdata = training_data_select(oj, marker, celltypes, cell_types,
                                                                           np.arange(len(cell_types)))

viz_training_data(oj, tot_lab, tot_ideal_ind, cell_types, oj.obsm['X_umap'], sns.color_palette('tab20', 8),
                  'Training/validation sets (~15%)', (6, 5), 0.75)

# FEEDFORWARD NEURAL NETWORK FOR CELL TYPE ANNOTATION, VISUALIZATION
learning_rate = 0.025  # altering learning rate to change how much neural net can adjust during each training epoch
training_epochs = 500
batch_size = 100
display_step = 5

# using aggregate data for training to bolster cell type abundances in training sets
tot_lab_onehot = one_hot_encode(tot_lab)
all_train_ind = np.array([])
ideal_ = np.argmax(tot_lab_onehot, axis=1)
train_split = 0.5
for k in np.unique(ideal_):
    all_ind = np.where(ideal_ == k)[0]  # randomly select half for training, other half goes to validation
    train_ind = np.random.choice(all_ind, round(train_split*len(all_ind)), replace=False)
    all_train_ind = np.concatenate((all_train_ind, train_ind))

total_predicted_lab, tot_prob, colorm, pred = cell_type_classifier(tot_lab_onehot, tot_traindata,
                                                                   tot_testdata,
                                                                   all_train_ind,
                                                                   learning_rate, training_epochs, batch_size,
                                                                   display_step)

# reordering cell type labels and filtering by probability
total_lab, total_prob = process_label(tot_prob, tot_lab, total_predicted_lab, tot_ideal_ind, oj, 0.9)

# storing cell type labels as metadata
oj.obs['nn_90'] = total_lab
oj.obs['nn_90'] = oj.obs['nn_90'].map(celltype_map)

# VIZ data by labeled cell type
sc.pl.umap(oj, color='nn_90', s=30)

# write to updated single cell object
oj.write('/Users/katebridges/Downloads/OJ_24_48_woundbeds_annotated.h5ad')
