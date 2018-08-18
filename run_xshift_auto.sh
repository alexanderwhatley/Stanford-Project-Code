#!/bin/bash

#SBATCH -n 32                            #Number of cores
#SBATCH -N 1                            #Run on 1 node
#SBATCH --mem=12000                       #Memory per cpu in MB (see also --mem)

#SBATCH -t 15:00:00                     #Runtime in minutes
#SBATCH -p serial_requeue               #Partition to submit to

cd ${1}
cp ../../${2} .
mv ${2} fcsFileList.txt
cp ../../importConfig.txt .
echo ${1}
module load centos6/0.0.1-fasrc01
module load java/1.8.0_45-fasrc01
java -Xmx10G -cp "../../VorteX.jar" standalone.Xshift auto