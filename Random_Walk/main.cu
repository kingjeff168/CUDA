/*
Last Date Modified: 1/9/2024
*/
#include <iostream>
#include <ctime>
#include <sstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>


using namespace std;

// This function is to calculate the distance of random walk from the origin.walkers
__global__ void randomWalks(float * result, unsigned int seed, long walkerSteps, long walkers) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state; // cuda random state
    curand_init(seed, tid, 0, &state);
    int tempX = 0, tempY = 0;
    float distanceTemp = 0;

    // 0~0.25 for left, 0.25 ~ 0.5 for down, 0.5 ~ 0.75 for up, 0.75 ~ 1 for right
    if (tid < walkers)
    {
        for (int i = 0; i < walkerSteps; ++i) 
        {
            float rand = curand_uniform(&state);

            if ((rand >= 0) && (rand < 0.25))
            {
                tempX -= 1;
            }
            else if ((rand >= 0.25) && (rand < 0.5))
            {
                tempY -= 1;
            }
            else if ((rand >= 0.5) && (rand < 0.75))
            {
                tempY += 1;
            }
            else
            {
                tempX += 1;
            }       
        }
        distanceTemp = sqrtf((tempX * tempX) + (tempY * tempY));
        result[tid] = distanceTemp;
    }
}

//**************** Warm Up Function***********************
void warmUp(long grid_size, int block_size, long walkers, long walkerSteps)
{
    float *distanceHost1;
    float *distanceDevice1;

    // Allocate host memory
    distanceHost1 = (float*)malloc(sizeof(float) * grid_size * block_size);

    // Allocate device memory
    cudaMalloc((void**)&distanceDevice1, sizeof(float) * grid_size * block_size);

    // Executing kernel 
    randomWalks<<<grid_size, block_size>>>(distanceDevice1, time(NULL), walkerSteps, walkers);
    
    // Transfer data back to host memory
    cudaMemcpy(distanceHost1, distanceDevice1, sizeof(float) * grid_size * block_size, cudaMemcpyDeviceToHost);

    // Add the distance of all walkers
    double averageDistance1 = 0;
    for (int i = 0; i < (grid_size * block_size); ++i) 
    {
        averageDistance1 += distanceHost1[i];
    }  

    // Deallocate device memory
    cudaFree(distanceDevice1);

    // Deallocate host memory
    free(distanceHost1);
}

//**************** Normal CUDA Memory Allocation Function***********************
void normalMemoryAllocation(long grid_size, int block_size, long walkers, long walkerSteps)
{
    float *distanceHost1;
    float *distanceDevice1;
    chrono::high_resolution_clock::time_point startTimer1;
    chrono::high_resolution_clock::time_point stopTimer1; 

    // Allocate host memory
    startTimer1 = chrono::high_resolution_clock::now();
    distanceHost1 = (float*)malloc(sizeof(float) * grid_size * block_size);

    // Allocate device memory
    cudaMalloc((void**)&distanceDevice1, sizeof(float) * grid_size * block_size);

    // Executing kernel 
    randomWalks<<<grid_size, block_size>>>(distanceDevice1, time(NULL), walkerSteps, walkers);
    
    // Transfer data back to host memory
    cudaMemcpy(distanceHost1, distanceDevice1, sizeof(float) * grid_size * block_size, cudaMemcpyDeviceToHost);

    // Add the distance of all walkers
    double averageDistance1 = 0;
    for (int i = 0; i < (grid_size * block_size); ++i) 
    {
        averageDistance1 += distanceHost1[i];
    }  

    // Deallocate device memory
    cudaFree(distanceDevice1);

    // Deallocate host memory
    free(distanceHost1);

    // Calculate the average distance and total calculation time of the random walk 
    stopTimer1 = chrono::high_resolution_clock::now();
    auto totalTime1 = chrono::duration_cast<chrono::microseconds>(stopTimer1 - startTimer1);
    cout << "Normal CUDA memory Allocation:" << endl;
    cout << "    Time to calculate(microsec): "<< totalTime1.count() << endl;
    averageDistance1 = averageDistance1 / walkers;
    cout << "    Average distance from origin: "<< averageDistance1 << endl;
}

//**************** Pinned CUDA Memory Allocation Function************************
void pinnedMemoryAllocation(long grid_size, int block_size, long walkers, long walkerSteps)
{
    // host arrays
    float *distanceHost2;
    
    // device array
    float *distanceDevice2;

    chrono::high_resolution_clock::time_point startTimer2;
    chrono::high_resolution_clock::time_point stopTimer2; 

    // allocate and initialize
    startTimer2 = chrono::high_resolution_clock::now();

    cudaMallocHost((void**)&distanceHost2, sizeof(float) * grid_size * block_size); // host pinned
    cudaMalloc((void**)&distanceDevice2, sizeof(float) * grid_size * block_size);   // device

    // Executing kernel 
    randomWalks<<<grid_size, block_size>>>(distanceDevice2, time(NULL), walkerSteps, walkers);
    
    // Transfer data back to host memory
    cudaMemcpy(distanceHost2, distanceDevice2, sizeof(float) * grid_size * block_size, cudaMemcpyDeviceToHost);

    // Add the distance of all walkers
    double averageDistance2 = 0;
    for (int i = 0; i < (grid_size * block_size); ++i) 
    {
        averageDistance2 += distanceHost2[i];
    }

    // Free memory
    cudaFree(distanceDevice2);
    cudaFreeHost(distanceHost2);

    // Calculate the average distance and total calculation time of the random walk 
    stopTimer2 = chrono::high_resolution_clock::now();
    auto totalTime2 = chrono::duration_cast<chrono::microseconds>(stopTimer2 - startTimer2);
    cout << "Pinned CUDA memory Allocation:" << endl;
    cout << "    Time to calculate(microsec): "<< totalTime2.count() << endl;
    averageDistance2 = averageDistance2 / walkers;
    cout << "    Average distance from origin: "<< averageDistance2 << endl;
}

//**************** Managed CUDA Memory Allocation Function**************************
void managedMemoryAllocation(long grid_size, int block_size, long walkers, long walkerSteps)
{
    // Allocate Unified Memory - accessible from CPU or GPU
    float *distanceDevice3;
    chrono::high_resolution_clock::time_point startTimer3;
    chrono::high_resolution_clock::time_point stopTimer3; 

    // allocate and initialize
    startTimer3 = chrono::high_resolution_clock::now();
    //distanceHost3 = (float*)malloc(sizeof(float) * NUM_BLOCKS * THREADS_PER_BLOCK);
    cudaMallocManaged(&distanceDevice3, grid_size * block_size * sizeof(float));


    // Run kernel on the GPU
    randomWalks<<<grid_size, block_size>>>(distanceDevice3, time(NULL), walkerSteps, walkers);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    
    // Add the distance of all walkers
    double averageDistance3 = 0;
    for (int i = 0; i < (grid_size * block_size); ++i) 
    {
        averageDistance3 += distanceDevice3[i];
    }


    // Free memory
    cudaFree(distanceDevice3);

    // Calculate the average distance and total calculation time of the random walk 
    stopTimer3 = chrono::high_resolution_clock::now();
    auto totalTime3 = chrono::duration_cast<chrono::microseconds>(stopTimer3 - startTimer3);
    cout << "Managed CUDA memory Allocation:" << endl;
    cout << "    Time to calculate(microsec): "<< totalTime3.count() << endl;
    averageDistance3 = averageDistance3 / walkers;
    cout << "    Average distance from origin: "<< averageDistance3 << endl;
}

// Check whether arguments at the command line are posituve integer
bool isPositiveInteger(const string& str) 
{
    // This for-loop is to check if the input is an integer.
    for (char c : str) 
    {
        if (!isdigit(c)) 
        {
            return false;
        }
    }
    return true;
}


int main(int argc, char* argv[])
{
    // Set the default value of the number of walkers and the number of steps.
    long walkers = 1000;
    long walkerSteps = 1000;

    // Determine whether the user assign the number of steps and the number of walkers.
    for (int i = 1; i < argc; i++)
    {
        if (argc > 1)
        {
            string arg = argv[i];
            if ((arg == "-W") && ((i + 1) < argc))
            {
                int i2 = i + 1;

                if (isPositiveInteger(argv[i2]))
                {
                    walkers = stol(argv[i2]);
                }
            }
            else if (((arg == "-I") && ((i + 1) < argc)) || ((arg == "-l") && ((i + 1) < argc)))
            {
                int i3 = i + 1;
                if (isPositiveInteger(argv[i3]))
                {
                    walkerSteps = stol(argv[i3]);
                }
            }
        }
    }

    int block_size = 120;
    long grid_size = ((walkers + block_size) / block_size);

    // Warm-up function for the initialization.
    warmUp(grid_size, block_size, walkers, walkerSteps);

    // Implement three memory models for the random walk.
    normalMemoryAllocation(grid_size, block_size, walkers, walkerSteps);
    pinnedMemoryAllocation(grid_size, block_size, walkers, walkerSteps);
    managedMemoryAllocation(grid_size, block_size, walkers, walkerSteps);
    cout << "Bye" << endl;

    return 0;
}