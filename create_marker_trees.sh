#!/bin/bash

#SBATCH -n 2                            #Number of cores
#SBATCH -N 1                            #Run on 1 node
#SBATCH --mem=8000                       #Memory per cpu in MB (see also --mem)

#SBATCH -t 2:00:00                     #Runtime in minutes
#SBATCH -p serial_requeue               #Partition to submit to

python create_marker_trees.py ${1} 