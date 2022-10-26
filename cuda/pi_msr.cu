/*

This program will numerically compute the integral of

                  4/(1+x*x)

from 0 to 1.  The value of this integral is pi -- which
is great since it gives us an easy way to check the answer.

History: Written by Tim Mattson, 11/1999.
         Modified/extended by Jonathan Rouzaud-Cornabas, 10/2022
*/

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <omp.h>
#include <fstream>

__global__ void calculate_pi(float *partial_pi, int n, int threads) {  
  extern __shared__ float partial_sum[];
  int tid = threadIdx.x;

  partial_sum[tid] = 0.0;

  for (long i = tid; i < n; i += threads) {
    partial_sum[tid] += partial_pi[i];
  }

  int batch = (int) threads / 2;

  __syncthreads();
  while (batch > 0) {
    if (tid < batch) partial_sum[tid] += partial_sum[tid + batch];
    __syncthreads();
    batch = (int) batch / 2;
  }

  if (tid == 0) partial_pi[0] = partial_sum[0];
}

__global__ void calculate_partial_pi(float *partial_pi, int n, int block_steps, int thread_steps, double step, long num_steps)
{
  extern __shared__ float partial_sum[];
  float sum = 0.0;  
  double x = 0.0;

  int tid = threadIdx.x;
  int bid = blockIdx.x; 

  long block_start = 1 + bid * block_steps;

  long thread_start = block_start + thread_steps * tid;
  long thread_end = thread_start + thread_steps;

  partial_sum[tid] = 0;

  for (long i = thread_start; i < thread_end; i++)
  {
    if (i > num_steps) {
      break;
    };
    x = (i - 0.5) * step;
    sum += 4.0 / (1.0 + x * x);
  }

  partial_sum[tid] = sum;

  int batch = (int) block_steps / (2 * thread_steps);

  __syncthreads();
  while (batch > 0) {
    if (tid < batch) partial_sum[tid] += partial_sum[tid + batch];
    __syncthreads();
    batch = (int) batch / 2;
  }

  if (tid == 0) partial_pi[bid] = partial_sum[0];
}

int main(int argc, char **argv)
{
  long num_steps = 10000000;
  int threads = 2;
  int thread_steps = 2;

  // Read command line arguments.
  for (int i = 0; i < argc; i++)
  {
    if ((strcmp(argv[i], "-N") == 0) || (strcmp(argv[i], "-num_steps") == 0))
    {
      num_steps = atol(argv[++i]);
      printf("  User num_steps is %ld\n", num_steps);
    }
    if ((strcmp(argv[i], "-t") == 0) || (strcmp(argv[i], "-threads") == 0))
    {
      threads = atol(argv[++i]);
    }
    if ((strcmp(argv[i], "-ts") == 0) || (strcmp(argv[i], "-thread_steps") == 0))
    {
      thread_steps = atol(argv[++i]);
    }
    else if ((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "-help") == 0))
    {
      printf("  Pi Options:\n");
      printf("  -num_steps (-N) <int>:      Number of steps to compute Pi (by default 100000000)\n");
      printf("  -help (-h):            print this message\n\n");
      exit(1);
    }
  }

  struct timeval begin, end;
  float *partial_pi, *d_partial_pi; 
  double pi, time = 0.0;
  const int block_steps = thread_steps * threads;
  const int N = ceil(num_steps / block_steps);
  const int blocks = N;
  const size_t block_mem_size = sizeof(float) * threads;
  double step = 1.0 / (double) num_steps;

  gettimeofday(&begin, NULL);

  // Allocate host memory
  partial_pi = (float*)malloc(sizeof(float) * num_steps);

  // Allocate device memory
  cudaError_t err = cudaMalloc((void**)&d_partial_pi, sizeof(float) * num_steps);
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n" , cudaGetErrorString(err));
  }

  // Call device to calculate partial pi (reduction 1)
  calculate_partial_pi<<<blocks, threads, block_mem_size>>>(d_partial_pi, N, block_steps, thread_steps, step, num_steps);

  cudaDeviceSynchronize();

  // Call device to calculate pi (reduction 2)
  calculate_pi<<<1, threads, block_mem_size>>>(d_partial_pi, N, threads);

  // Transfer data back to host memory
  cudaMemcpy(partial_pi, d_partial_pi, sizeof(float) * N, cudaMemcpyDeviceToHost);

  // Get final result
  pi = partial_pi[0] * step;

  // Deallocate device memory
  cudaFree(d_partial_pi);

  // Deallocate host memory
  free(partial_pi);
  
  // Metrics
  gettimeofday(&end, NULL);

  time = 1.0 * (end.tv_sec - begin.tv_sec) +
          1.0e-6 * (end.tv_usec - begin.tv_usec);

  std::ofstream file;
  file.open("pi_msr.csv", std::ios_base::app);

  file << num_steps << "," << blocks << "," << threads << "," << thread_steps << "," << pi << "," << time << std ::endl;

  file.close();
}