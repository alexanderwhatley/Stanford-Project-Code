import sys
import shutil
import numpy as np
import pandas as pd
from utils import *

if __name__ == '__main__':
    path_ind = int(sys.argv[1])
    tree = pickle.load(open('tree_combined_for_html.pkl', 'rb'))
    path = get_path(tree, path_ind)
    depth = get_tree_depth(tree)
    if not os.path.isdir('Node_Data/Path_{}'.format(path_ind)):
        os.mkdir('Node_Data/Path_{}'.format(path_ind))
    for node_ind, node in enumerate(path):
        node.coords.to_csv('Node_Data/Path_{}/Path_{}_Node_{}.csv'.format(path_ind, path_ind, node_ind), index=False)
    for i in range(node_ind+1, depth):
        coords = pd.DataFrame(columns=['X', 'Y', 'Z', 'sample'])
        coords.to_csv('Node_Data/Path_{}/Path_{}_Node_{}.csv'.format(path_ind, path_ind, i), index=False)
        
    shutil.make_archive('Node_Data/Path_{}'.format(path_ind), 'gztar', 'Node_Data/Path_{}'.format(path_ind))
    shutil.rmtree('Node_Data/Path_{}'.format(path_ind)) 