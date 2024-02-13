This is a C++ CUDA program to simulate a 2D random walk. 

It simulates a large number of walkers taking steps either north, south, east, or west on a grid, 
and calculate the average distance they travel from the origin. 
I use different memory models, including cudaMalloc, cudaMallocHost, and cudaMallocManaged, to perform the calculations.