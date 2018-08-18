
from collections import defaultdict, Counter 
import colorsys
import itertools
import pickle 
import io, os, sys, types

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy.spatial import distance, ConvexHull
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
import fcsparser 
from ete3 import Tree, TreeNode, TreeStyle, TextFace

from utils import *

if __name__ == '__main__':
    path_ind = int(sys.argv[1])
    tree = pickle.load(open('tree_combined_for_html.pkl', 'rb'))
    depth = get_tree_depth(tree)
    path = get_path(tree, path_ind)
    plot_path_clusters(path, path_ind, depth)