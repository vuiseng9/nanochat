#!/usr/bin/env bash

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/root/work/nanochat-cache"
mkdir -p $NANOCHAT_BASE_DIR

source .venv/bin/activate

# python -m nanochat.report reset

export WANDB_API_KEY=${1:?"Error: No WANDB_API_KEY, pls provide"}

GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | awk '{print $2}')
DATE=$(date +"%y%m%d")
NLAYER=20
NGPU=8
PREC=bf16
    # --prec=${PREC} \

python -m torch.distributed.run --standalone --nproc_per_node=${NGPU} \
    -m scripts.mid_train -- \
    --run=${DATE}__${PREC}_d${NLAYER}_mid_train_${GPU_MODEL}x${NGPU}

