CMD='python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 cluster.py'
echo $CMD
$CMD
