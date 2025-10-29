#!/usr/bin/env bash

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/root/work/nanochat-cache"
mkdir -p $NANOCHAT_BASE_DIR

source /opt/venv/bin/activate

python -m nanochat.report reset

export WANDB_API_KEY=${1:?"Error: No WANDB_API_KEY, pls provide"}

GPU_MODEL=$(rocm-smi --showproductname | grep --color=auto "Card Series" | head -n1 | awk -F' ' '{print $7}')
DATE=$(date +"%y%m%d")
NLAYER=20
NGPU=8
PREC=bf16
python -m torch.distributed.run --standalone --nproc_per_node=${NGPU} \
    -m scripts.base_train -- \
    --depth=${NLAYER} \
    --prec=${PREC} \
    --run=${DATE}__${PREC}_d${NLAYER}_base_train_${GPU_MODEL}x${NGPU}

