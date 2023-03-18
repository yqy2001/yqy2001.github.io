# Fairscale Usage

Reference: https://fairscale.readthedocs.io/en/latest/deep_dive/oss_sdp_fsdp.html

Model states: optimizer states, gradients, parameters.

Optimizers such as Adam usually maintain momentum, variance, parameters and gradients all in FP32 precision.

### Optimizer State Sharding (OSS, ZeRO-1)

OSS partitions the model optimization step among different ranks, so that each of them is only in charge of updating a unique shard of the model. Specifically, OSS involves breaking up the optimizer states into smaller pieces and distributing those pieces across multiple processing units or 'ranks'. The global systems then only keep one copy of optimizer states accross different ranks during training.

Implementation:

The wrapping of the optimizer is a one-line non intrusive change that provides memory savings.

### Sharded Data Parallel (SDP, ZeRO-2)

While OSS solved the optimizer redundancy problem, the data parallel training steps mentioned above revealed a duplication of computation during gradient aggregation, as well as additional memory being used for gradient discarding.

What's been reduced: allreduce to reduce, 1. less communication; 2. less memory.

### Fully Sharded Data Parallel (FSDP, ZeRO-3)

data parallel ranks are responsible for a shard of the model parameters.

Note that FSDP is still **data parallel**

Implementation:

autowrap

If combined with activation checkpointing, it is preferable to use FSDP(checkpoint_wrapper(module)) over checkpoint_wrapper(FSDP(module)). The latter will result in more communication and will be slower.