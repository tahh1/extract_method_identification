#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=0:30:0   
#SBATCH --mail-user=indranil.palit@dal.ca
#SBATCH --mail-type=ALL

module purge

module load java/17.0.2
module load python/3.10


export JAVA_TOOL_OPTIONS="-Xms256m -Xmx5g"
export PATH=$PATH:$refresearch/executable/RefactoringMiner/bin

cd $refresearch
source res_venv/bin/activate

python data_collector.py /home/ip1102/projects/def-tusharma/ip1102/Ref-Res/Research/dummy.csv