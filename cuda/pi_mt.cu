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

__global__ void calculate_pi(float *partial_pi_blocks, float *partial_pi_threads, int n, int block_steps, int thread_steps, double step, long num_steps)
{
  float x = 0.0;
  int threads = blockDim.x;
  int bid = blockIdx.x;
  int tbid = threadIdx.x; // thread id considering its block

  float sum = 0.0;
  long block_start = 1 + bid * block_steps;

  long thread_start = block_start + thread_steps * tbid;
  long thread_end = thread_start + thread_steps;

  for (long i = thread_start; i < thread_end; i++)
  {
    if (i > num_steps) {
      break;
    };
    x = (i - 0.5) * step;
    sum += 4.0 / (1.0 + x * x);
  }

  partial_pi_threads[tbid] = sum;

  __syncthreads();

  if (tbid == 0) {
    for (int j = 0; j < threads; j++) {
      partial_pi_blocks[bid] += partial_pi_threads[j]; 
    }
  }
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
  float *partial_pi_blocks, *d_partial_pi_blocks, *partial_pi_threads, *d_partial_pi_threads; 
  double pi, time = 0.0;
  const int block_steps = thread_steps * threads;
  const int N = ceil(num_steps / block_steps);
  const int blocks = N;
  double step = 1.0 / (double) num_steps;

  gettimeofday(&begin, NULL);

  // Allocate host memory
  partial_pi_blocks = (float*)malloc(sizeof(float) * blocks);
  partial_pi_threads = (float*)malloc(sizeof(float) * threads);

  // Allocate device memory
  cudaMalloc((void**)&d_partial_pi_blocks, sizeof(float) * blocks);
  cudaMalloc((void**)&d_partial_pi_threads, sizeof(float) * threads);

  // Call device
  calculate_pi<<<blocks, threads>>>(d_partial_pi_blocks, d_partial_pi_threads, N, block_steps, thread_steps, step, num_steps);

  cudaDeviceSynchronize();

  // Transfer data back to host memory
  cudaMemcpy(partial_pi_blocks, d_partial_pi_blocks, sizeof(float) * blocks, cudaMemcpyDeviceToHost);

  // Get final result
  for(int i = 0; i < blocks; i++){
    pi += partial_pi_blocks[i];
  }
  pi *= step;

  // Deallocate device memory
  cudaFree(d_partial_pi_blocks);
  cudaFree(d_partial_pi_threads);

  // Deallocate host memory
  free(partial_pi_blocks);
  free(partial_pi_threads);
  
  // Metrics
  gettimeofday(&end, NULL);

  time = 1.0 * (end.tv_sec - begin.tv_sec) +
          1.0e-6 * (end.tv_usec - begin.tv_usec);

  std::ofstream file;
  file.open("pi_mt.csv", std::ios_base::app);

  file << num_steps << "," << blocks << "," << threads << "," << thread_steps << "," << pi << "," << time << std::endl;

  file.close();
}