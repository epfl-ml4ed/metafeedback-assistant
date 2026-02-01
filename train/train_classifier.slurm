#!/bin/bash
#SBATCH --job-name=hf_unified_train
#SBATCH --output=logs/hf_unified_%j.out
#SBATCH --error=logs/hf_unified_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# --- Environment setup ---
export HF_HOME=/scratch/izar/pitu/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

module purge
module load gcc/11.3.0 cuda/12.1.1

# Activate conda environment
source /scratch/izar/pitu/miniconda3/etc/profile.d/conda.sh
conda activate hf-train

# Move into project folder
cd /home/pitu/metafeedback_feedback_on_feedback

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "============================================"
which python
echo "============================================"

# Run with default 
/scratch/izar/pitu/miniconda3/envs/hf-train/bin/python classifier/new/trainer_unified_bert.py

echo "Python path: $(which python)"
python -c "import sys; print('Python version:', sys.version)"
python -c "import transformers; print('Transformers version in job:', __import__('transformers').__version__)"

# Run with a YAML config 
#/scratch/izar/pitu/miniconda3/envs/hf-train/bin/python classifier/new/trainer_unified.py --config configs/experiment_02.yaml

echo "--------------------------------------------"
echo " Job $SLURM_JOB_ID finished at: $(date)"
echo "--------------------------------------------"
