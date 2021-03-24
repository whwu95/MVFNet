#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
PORT=${PORT:-28888}
$PYTHON -m torch.distributed.launch --nproc_per_node=$2 --master_port=$PORT train_recognizer.py $1 \
--launcher pytorch  --validate  ${@:3}

