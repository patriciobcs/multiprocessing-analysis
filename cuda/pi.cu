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

static long num_steps = 10000000;
static long num_threads = 2;
double step;

__global__ void calculate_pi(double *partial_pi, int n, double slice_size, double step)
{
  double x = 0.0;
  for (int j = 0; j < n; j++)
  {
    double partial_sum = 0.0;
    int start = 1 + j * slice_size;
    int end = start + slice_size;

    for (int i = start; i <= end; i++)
    {
      x = (i - 0.5) * step;
      partial_sum += 4.0 / (1.0 + x * x);
    }
    partial_pi[j] += partial_sum;
  }
}

int main(int argc, char **argv)
{
  int 
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
      num_threads = atol(argv[++i]);
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
  double *partial_pi, *d_partial_pi; 
  double pi, time = 0.0;
  const int N = 100; // batches
  const int slice_size = num_steps / N;

  step = 1.0 / (double)num_steps;

  gettimeofday(&begin, NULL);

  // Allocate host memory
  partial_pi = (double*)malloc(sizeof(double) * N);

  // Allocate device memory
  cudaMalloc((void**)&d_partial_pi, sizeof(double) * N);

  calculate_pi<<<1, 1>>>(d_partial_pi, N, slice_size, step);

  // Transfer data back to host memory
  cudaMemcpy(partial_pi, d_partial_pi, sizeof(double) * N, cudaMemcpyDeviceToHost);

  for(int i = 0; i < N; i++){
    pi += partial_pi[i];
  }
  pi *= step;

  // Deallocate device memory
  cudaFree(d_partial_pi);

  // Deallocate host memory
  free(partial_pi);
  
  gettimeofday(&end, NULL);

  time = 1.0 * (end.tv_sec - begin.tv_sec) +
          1.0e-6 * (end.tv_usec - begin.tv_usec);

  std::ofstream file;
  file.open("pi.csv", std::ios_base::app);

  file << num_steps << "," << num_threads << "," << pi << "," << time << std ::endl;

  file.close();
}