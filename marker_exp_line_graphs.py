from collections import defaultdict, Counter 
import colorsys
import itertools
import pickle 
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from PyPDF2 import PdfFileWriter, PdfFileReader, PdfFileMerger
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from IPython.display import display, HTML

import ot
from scipy.spatial import distance, ConvexHull
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
import fcsparser 
from ete3 import Tree, TreeNode, TreeStyle, TextFace, add_face_to_node

from utils import *

def plot_line_graphs(tree, path, path_ind):
    output_dir = 'Marker_Expression_Line_Graphs/Path_{}'.format(path_ind)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    supplementary_data = pd.read_csv('../Suppl.Table2.CODEX_paper_MRLdatasetexpression.csv')
    supplementary_data.rename(columns={'X.X': 'X', 'Y.Y': 'Y', 'Z.Z': 'Z'}, inplace=True)
    supplementary_data['CD45_int'] = supplementary_data['CD45'].astype(int)
    ids_to_names = pd.read_csv('ClusterIDtoName.txt', sep='\t')
    cell_lines = list(ids_to_names['ID'].values)
    ids_to_names = dict(zip(ids_to_names['ID'].values, ids_to_names['Name'].values))
    # remove dirt from supplementary data 
    supplementary_annotations = pd.read_excel('../Suppl.Table2.cluster annotations and cell counts.xlsx')
    dirt = supplementary_annotations.loc[supplementary_annotations['Imaging phenotype (cell type)'] == 'dirt', 
                                         'X-shift cluster ID']
    supplementary_data = supplementary_data[~supplementary_data['Imaging phenotype cluster ID'].isin(dirt)]
    supplementary_data['sample'] = supplementary_data['sample_Xtile_Ytile'].apply(lambda x: x.split('_')[0])
    suppl_converted = convert_coordinates(supplementary_data)[['X', 'Y', 'Z', 'sample'] + marker_cols]
    
    depth = get_tree_depth(tree)
    for node_ind, node in enumerate(path[1:]):
        overlap = pd.merge(suppl_converted, node.coords, how='inner', on=['X', 'Y', 'Z', 'sample'])
        overlap = overlap[marker_cols].mean()
        overlap = [overlap[marker_col] for marker_col in marker_cols]
        plt.plot(np.arange(len(marker_cols)), overlap)
        plt.xticks(np.arange(len(marker_cols)), marker_cols, rotation='vertical')
        plt.savefig(output_dir + '/Path_{}_Node_{}_Avg_Exp.pdf'.format(path_ind, node_ind))
        plt.clf()
    for i in range(node_ind+1, depth):
        plt.plot([], [])
        plt.xticks(np.arange(len(marker_cols)), marker_cols, rotation='vertical')
        plt.savefig(output_dir + '/Path_{}_Node_{}_Avg_Exp.pdf'.format(path_ind, i))
        plt.clf()  
        
    shutil.make_archive(output_dir, 'gztar', output_dir)
    shutil.rmtree(output_dir) 
    
if __name__ == '__main__':
    path_ind = int(sys.argv[1])
    tree = pickle.load(open('tree_combined_for_html.pkl', 'rb'))
    path = get_path(tree, path_ind)
    plot_line_graphs(tree, path, path_ind)