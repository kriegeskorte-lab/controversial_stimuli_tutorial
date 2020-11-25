#!/bin/sh
#
#
#SBATCH --account=nklab
#SBATCH --job-name=controversial    # The job name.
#SBATCH --gres=gpu:1
#SBATCH --cores=2
#SBATCH --mem=16GB
#SBATCH --error="slurm-%A_%a.err"
#SBATCH --time=4-0:0

module load anaconda3-2019.03 

. activate controversial_stimuli_tutorial_env

python -u batch_optimize.py
echo "python script finished."

# End of script

