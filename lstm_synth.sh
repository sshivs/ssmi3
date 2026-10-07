#!/bin/bash
#SBATCH -J synth_lstm
#SBATCH -o /home/fareeds/logs/%j.out      # Standard output
#SBATCH -e /home/fareeds/logs/%j.err      # Standard error
#SBATCH -p mit_normal                     # mit_normal_gpu
#SBATCH --mail-user=fareeds@mit.edu       # Email notifications
#SBATCH --mail-type=ALL                   # Send email on all events
#SBATCH --mem=64G                         # Request 64 GB of memory (smaller for synthetic data)
#SBATCH --time=4:00:00                    # Set a time limit of 4 hours (smaller dataset)

# Ensure the logs directory exists
mkdir -p /orcd/home/002/fareeds/logs

# Activate virtual environment
conda init
conda activate cvi4
which conda

# Change to the working directory
cd /orcd/home/002/fareeds/constrained-inference/advi_ssm_mimic4

# Load necessary modules
# module load cuda/11.1

# Run the Python script
python -u run_experiment.py -c config.synthetic_ext_lstm

# Print completion message and date
echo "Run completed at:- "
date


