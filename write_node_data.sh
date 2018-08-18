#!/bin/bash

#SBATCH -n 2                            #Number of cores
#SBATCH -N 1                            #Run on 1 node
#SBATCH --mem=8000                       #Memory per cpu in MB (see also --mem)

#SBATCH -t 2:00:00                     #Runtime in minutes
#SBATCH -p serial_requeue               #Partition to submit to

python write_node_data.py ${1} 