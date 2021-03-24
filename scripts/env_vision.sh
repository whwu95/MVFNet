# CONDA_PATH=/ssd1/vis/wuwenhao/software/miniconda3_cuda10_cluster  # cuda10 pytorch1.5
# source $CONDA_PATH/bin/activate
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PATH/x86_64-conda_cos6-linux-gnu/sysroot/lib/
# export LD_LIBRARY_PATH=$CONDA_PATH/x86_64-conda_cos6-linux-gnu/sysroot/lib/

# 集群的LD_LIBRARY_PATH里面太多脏东西，比如NCCL version不匹配
# export LD_LIBRARY_PATH=/home/opt/nvidia_lib/:$CONDA_PATH/x86_64-conda_cos6-linux-gnu/sysroot/lib/


export PY37_HOME='/ssd1/vis/wuwenhao/software/miniconda3_cuda10_cluster'                                                                                                                             
export PATH=${PY37_HOME}/bin:$PATH                                                                                                                                                                 
export LD_LIBRARY_PATH=${PY37_HOME}/lib/:${LD_LIBRARY_PATH}     