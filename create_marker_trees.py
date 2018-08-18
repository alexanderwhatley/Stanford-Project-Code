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
from ete3 import Tree, TreeNode, TreeStyle, TextFace

from utils import *

def get_node_markers(node, marker, suppl_converted):
    node_marker = pd.merge(node.coords, suppl_converted, how='inner', on=['X', 'Y', 'Z', 'sample'])
    marker_avg = node_marker[marker].mean()
    return marker_avg

def plot_marker_tree(tree, marker, resize_nodes=False, save=True):
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
    suppl_converted = convert_coordinates(supplementary_data)[['X', 'Y', 'Z', 'sample', marker]]
    
    new_tree = TreeNode(name = tree.name)
    new_tree.img_style['size'] = 1 if resize_nodes else 10
    new_tree.img_style['fgcolor'] = hls2hex(0, 0, 0)
    new_tree.img_style['shape'] = 'sphere'
    
    marker_avgs = []
    old_layer = [tree]
    new_layer = [new_tree]
    layer_num = 0
    while old_layer:
        next_old_layer, next_new_layer = [], []
        for ind, node in enumerate(old_layer):
            for child in node.children:
                next_old_layer.append(child)
                new_child = TreeNode(name = child.name)
                marker_avg = get_node_markers(child, marker, suppl_converted)
                new_child.add_features(marker_avg=marker_avg)
                marker_avgs.append(marker_avg)
                new_layer[ind].add_child(new_child)
                next_new_layer.append(new_child)
        old_layer = next_old_layer
        new_layer = next_new_layer
        layer_num += 1
        
    marker_min, marker_max = np.min(marker_avgs), np.max(marker_avgs)
    for node in new_tree.iter_descendants():
        norm_marker = (node.marker_avg - marker_min) / (marker_max - marker_min)
        node.add_features(marker_avg=norm_marker)
        node.add_features(color=hls2hex(0, norm_marker, norm_marker*0.5))
        
    for node in new_tree.iter_descendants():
        node.img_style['size'] = 1 + 10 * node.marker_avg if resize_nodes else 10
        node.img_style['fgcolor'] = node.color
        node.img_style['shape'] = 'sphere'
        
    ts = TreeStyle()
    ts.show_leaf_name = False
    ts.rotation = 90
    ts.title.add_face(TextFace(marker, fsize=20), column=0)
    save_dir = 'Marker_Trees' if resize_nodes else 'Marker_Trees_Same_Size'
        
    if save:
        new_tree.render(save_dir + '/marker_tree_{}.png'.format(marker), tree_style=ts)
    else:
        return new_tree.render('%%inline', tree_style=ts)
    
if __name__ == '__main__':
    marker = sys.argv[1]
    tree = pickle.load(open('tree_combined_for_html.pkl', 'rb'))
    plot_marker_tree(tree, marker, resize_nodes=False, save=True)
    #plot_marker_tree(tree, marker, resize_nodes=True, save=True)