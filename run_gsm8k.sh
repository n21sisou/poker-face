#!/bin/bash
#SBATCH --job-name=gsm8k
#SBATCH --output=logs/gsm8k_full_%j.out
#SBATCH --error=logs/gsm8k_full_%j.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-gpu=31 
#SBATCH --mem=32G
#SBATCH --export=ALL

# Optional: Email notifications (uncomment and add your email)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nsisouph@gmail.com


# Default model
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Number of samples (0 = all 1319 test examples)
N_SAMPLES=0

# Batch size (adjust based on GPU memory)
# For 3090 (24GB): batch_size=8-16
# For A100 (40/80GB): batch_size=16-32
BATCH_SIZE=16

# Max tokens for generation
MAX_TOKENS=2048

# Optional: use 8-bit quantization to save memory
USE_8BIT=""  # Set to "--load_in_8bit" to enable


echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "=========================================="

# Show GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Activate environment
source /Brain/private/n21sisouphanthong-vongkotrattana/repos/poker-face/.venv/bin/activate


python evaluate_gsm8k.py \
    --model "$MODEL" \
    --n_samples $N_SAMPLES \
    --batch_size $BATCH_SIZE \
    --max_tokens $MAX_TOKENS \
    $USE_8BIT

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
ls -lh results/gsm8k/