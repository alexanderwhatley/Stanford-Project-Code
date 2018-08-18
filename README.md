# Stanford-Project-Code

Make sure that you have Python 3 installed along with the packages numpy, pandas, matplotlib, seaborn, sklearn, pot, and ete3. 
Note that in order to run many of these scripts you will need access to a supercomputer cluster with SLURM installed. 

1. The file run_xshift.ipynb contains the code needed to run X-Shift on the FCS files.

2. utils.py contains most of the helper functions used inside the notebooks (such as calculating the metrics, 
preprocessing the data files, etc.)

3. In order to run many of the scripts, you need to have the serialized version of the tree object, called 
"tree_combined_for_html.pkl", as well as the supplementary data files "Suppl.Table2.cluster annotations and cell counts.xlsx
" and "Suppl.Table2.CODEX_paper_MRLdatasetexpression.csv" in the same directory as the scripts. These files can be found in the 
Google Drive folder. 
If you wish to run X-Shift, you also need to download the FCS Files folder, which can also be found in the Google Drive folder. 

4. The code to generate the images of the trees is found in generate_files.ipynb. In the section 'Write Tree Nodes to Files', 
we have the code for generating the 3x3 graphs for each sample and each path in the tree. The code in the sections 
'Plot Tree Where Color Depends on Marker Expression' and 'Draw Average Expression Line Charts for Each Node
' creates the colored trees based on marker expression and the line graphs. 

5. The file map_clusters_to_types.ipynb contains the run for preprocessing and running Vite to determine cell types. It contains
files that generate the input files to Vite, as well as generating the annotated trees with the predicted types. 
