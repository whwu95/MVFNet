#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
PORT=${PORT:-28890}

# run in node 1
$PYTHON -m torch.distributed.launch \
    --nproc_per_node=$2 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="10.127.20.17" \
    --master_port=$PORT \
    train_recognizer.py $1 \
    --launcher pytorch  --validate  ${@:3}

