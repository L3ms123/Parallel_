### Metrics

## Base version

| Version   | Seconds | Encerts |
| --------- | ------- | ------- |
| \*CPU     | 11.1    | 929     |
| GTX1080Ti | 8.35    | 928     |
| RTX2070   | 7.4     | 928     |
| RTX3080   | 6.4     | 928     |

\*CPU time was measured from aolin-login.

## Debug (Epoch = 1)

During development, it could be that execution times are quite big. We want to test our developments taking the less seconds possible. For that, we can temporally change epoch to 1.

| Version   | Seconds | Encerts\* |
| --------- | ------- | --------- |
| CPU       | 1.4     | 154       |
| GTX1080Ti | 1.3     | 146       |

\*The difference on "Encerts" is likely because floating sum operations are unordered in CUDA, resulting in slight difference on decimals (more info in: https://docs.nvidia.com/cuda/floating-point/index.html)

## How to improve?

To check the potential of the GPU, we can do a scalability test by increasing the number of neurons. This will increase the parallel work, but not necesarily means that the results will be better. Ideally increasing the number of neurons, we should reach a big number of encerts faster, for example, using less epoch. In that scenario, GPU will be much faster than CPU.

## Neurons: 1000x420x270 (V2)

Let's do a simple test: Increase the number of neurons by 2, from 500x210x135 to 1000x420x270. We hope this keeps the same or higher number of encerts, and ideally less epoch because we will have a faster case that works better in the GPU than in the CPU, since it paralelizes more data.

| Version | Encerts |
| ------- | ------- |
| CPU     | 757     |
| GPU     | 757     |

Encerts are the same between CPU and GPU, so GPU is still correct. Encerts are lower than before (757vs929). Let'ts see if we can use less epoch.

# Base loss

Epoch 0 Loss: 0.363908
Epoch 1 Loss: 0.357409
Epoch 2 Loss: 0.147380
Epoch 3 Loss: 0.061898
Epoch 4 Loss: 0.039837
Epoch 5 Loss: 0.026030
Epoch 6 Loss: 0.022047
Epoch 7 Loss: 0.016594
Epoch 8 Loss: 0.012993
Epoch 9 Loss: 0.011011

# V2 loss

Epoch 0 Loss: 0.392178
Epoch 1 Loss: 0.483458
Epoch 2 Loss: 0.262542
Epoch 3 Loss: 0.224882
Epoch 4 Loss: 0.233637
Epoch 5 Loss: 0.237597
Epoch 6 Loss: 0.244020
Epoch 7 Loss: 0.245151
Epoch 8 Loss: 0.252382
Epoch 9 Loss: 0.256206

Comparing the loss of both versions, V2 is much higher than Base. We can not use less epoch with these values.

Let's see if at least the GPU excels the CPU more than before.

\*Let's set that we use GTX1080Ti for all measurements, and the speedup is timeCPU/timeGPU.

| Version | TimeCPU(s) | TimeGPU(s) | Speedup |
| ------- | ---------- | ---------- | ------- |
| Base    | 11.1       | 8.35       | 1.3     |
| V2      | 28.2       | 8.9        | 3.2     |

Now we can see that the speedup increased significantly, (from 1.3 to 3.2, it is a 2.5x times increase) when increasing by 2 the neurons. This is more than a linear increase (2.5x increase in speedup against factor of 2), which states that the previous values were under-utilizing the GPU resources, maybe because were less closer to the power of 32 (warp size) or because of some under-usage of memory. Profilers like nvprof can be used to confirm this in case of wanting to explore more extensively the potential of the GPU, but alongside this, it is required an extra work of finding values that make worth to use the neuronal network.

# Conclusions

We have seen how much the GPU is faster than a single core-CPU. We tried a simple test of increasing the data by 2, but the encerts does not increase or remain equal. A possible solution is to review the formula that governs the neuronal network to find the optimal point, or in case of no having the formula, try much more tests, for example, configuring an script that runs many times the code with many different numbers of neurons, checking the epoch and the loss and taking the best value. This would be stopped at some time, and then plot the results to try to find a formula, and see if more neurons effectively increase the accuracy, or if we reached a dead point here.

Another possible solution is to try a different parallelization. For example, is there a way to parallelize all the neurons together? Or the patterns, and then use much more patterns? (For example, launch multiple neural networks in parallel, like one for each digit, letter of the alphabet, etc, or even use another dataset such as fingers).

All of these ideas represent different paths of researching. It is up to the reader to ask also if this is a worth path to follow, since the objective of this lab exercise is to learn how to parallelize, analyze results, and propose possible optimizations. For the part of the teaching, we like to conclude that we achieve the objective of showing a methodology of work.
