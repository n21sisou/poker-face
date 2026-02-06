#!/bin/bash

# GSM8K Evaluation Script
# Run DeepSeek-R1-Distill models on GSM8K benchmark

# Default model
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Number of samples (0 = all 1319 test examples)
N_SAMPLES=10

# Batch size (adjust based on GPU memory)
# For 3090 (24GB): batch_size=8-16
# For A100 (40/80GB): batch_size=16-32
BATCH_SIZE=8

# Max tokens for generation
MAX_TOKENS=2048

# Optional: use 8-bit quantization to save memory
USE_8BIT=""  # Set to "--load_in_8bit" to enable

echo "Running GSM8K evaluation..."
echo "Model: $MODEL"
echo "Samples: $N_SAMPLES"
echo "Batch size: $BATCH_SIZE"

python evaluate_gsm8k.py \
    --model "$MODEL" \
    --n_samples $N_SAMPLES \
    --batch_size $BATCH_SIZE \
    --max_tokens $MAX_TOKENS \
    $USE_8BIT

echo "Evaluation complete! Check results/gsm8k/ for output."
