#!/bin/bash
#SBATCH --job-name=ali_python_job       # Job name
#SBATCH --output=result_%j.out         # Output file (%j will append the job ID)
#SBATCH --error=error_%j.err           # Error file (%j will append the job ID)
#SBATCH --partition=high2              # Partition to submit to
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=1                     # Number of tasks (processes)
#SBATCH --cpus-per-task=12              # Number of CPU cores per task (adjust based on node specifications)
#SBATCH --mem=128G                      # Memory per node (adjust based on node specifications)
#SBATCH --time=12:00:00                # Time limit hrs:min:sec

module load conda

# Activate the Conda environment
source ~/.bashrc           # Ensure .bashrc is sourced
conda activate myenv       # Activate the myenv environment

echo "Program Started!"
# Run your Python script
/home/amehdiza/.conda/envs/myenv/bin/python /home/amehdiza/TextSumm/main.py
