# Plotting 3D and 2D graph
Visualize relations between power/clock/epoch_time data.

Input: data with multiple dimensions.

Output: 4 subgraphs. 1 for 3D data. Other 3 are for pairwise scatter graph. Pairwise scatter graph include fitting line.


# Known issue
gl2ps 1.3.8 contains one bug, which exported pictures dropped some elements on graph.


----------------------------
# Reports
![Power, Clock, and Time for each GPU](images/each_gpu.png)

1. Power consumption is nearly linear(sometimes qudratic) to GPU clock. So we can assume, if we limited clock, we can get reduced power.
2. GPU clock is strongly related to training time. Higher clock leads to less training time.
3. Using FP16 can reduce both power and time at the same time.
4. If GPU is power-limited, it would automatically reduce GPU clock, hence leads to long training time.

![GPU clock during training](images/gclk_160W_maxq.png)