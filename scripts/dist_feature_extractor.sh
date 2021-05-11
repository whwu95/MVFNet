#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}
# Using DistributedDataParallel
$PYTHON -m torch.distributed.launch --nproc_per_node=$3 feature_extractor.py $1 $2 --launcher pytorch ${@:4}
#  Using DataParallel
# $PYTHON test_recognizer.py $1 $2 --launcher none ${@:4}