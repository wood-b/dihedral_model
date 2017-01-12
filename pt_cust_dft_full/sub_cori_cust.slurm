#!/bin/bash -l

#SBATCH -p regular
#SBATCH --qos=premium
#SBATCH -N 2
#SBATCH -t 02:30:00
#SBATCH -J pt_dft_full

source activate bw_rubicon_cori 

NNODES=$SLURM_NNODES
cd $SLURM_SUBMIT_DIR

export QCSCRATCH=/dev/shm/eg_qchem
. /project/projectdirs/jcesr/qchem/qc44/cnl_qc441/dist/qcenv.sh

qcaux_setup start $NNODES
cat dft_input_list.txt | parallel -j $NNODES "cd {//}; fake_run_qchem.sh {%} -f {/} -e ; cd .."
qcaux_setup stop $NNODES

