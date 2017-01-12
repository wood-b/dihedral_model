#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J test_submit


module load python/2.7.10
module load numpy
module load scipy
module load gcc/5.1.0
source /global/homes/b/bwood/qchem_env/bin/activate 

NNODES=$SLURM_NNODES
cd $SLURM_SUBMIT_DIR

export QCSCRATCH=/dev/shm/eg_qchem
. /project/projectdirs/jcesr/qchem/qc43/cnl_qc432/dist/qcenv.sh

qcaux_setup start $NNODES
cat dft_input_list.txt | parallel -j $NNODES "cd {//}; fake_run_qchem.sh {%} -f {/} -e ; cd .."
qcaux_setup stop $NNODES

