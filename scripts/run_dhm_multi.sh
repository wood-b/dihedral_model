#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=12:00:00
#SBATCH --partition=matgen_debug
#SBATCH --account=matgen 
#SBATCH --job-name=dihedral_model_job

source dihedral/bin/activate

cd $SLURM_SUBMIT_DIR

srun -n16 --multi-prog /global/homes/b/bwood/dihedral/codes/dihedral_model/scripts/full_neutral_charged_n16.conf