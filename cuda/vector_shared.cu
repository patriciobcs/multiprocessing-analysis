/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <vector>

#include <fstream>
#include <cmath>

void checkSizes(long &N, long &M, long &S);

__global__ void calculate(float *d_partial_result, const float *d_A, const float *d_x, const float *d_y, int M, int N, int batch) {
  extern __shared__ float partial_mult[];
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  float mult = 0.0;

  int block_start = bid * M;

  int thread_start = tid * batch;
  int thread_end = thread_start + batch;

  for (int j = thread_start; j < thread_end; j++)
  {
    mult += d_A[block_start+j] * d_x[j];
  }

  partial_mult[tid] = mult * d_y[bid];
  
  int tbatch = blockDim.x; 

  __syncthreads();
  while (tbatch > 0) {
    if (tid < tbatch) partial_mult[tid] += partial_mult[tid + tbatch];
    __syncthreads();
    tbatch = (int) tbatch / 2;
  }

  if (tid == 0) d_partial_result[bid] = partial_mult[0];
}

int main(int argc, char *argv[])
{
  long N = -1;       // number of rows 2^12
  long M = -1;       // number of columns 2^10
  long S = -1;       // total size 2^22
  int num_threads = 1;

  // Read command line arguments.
  for (int i = 0; i < argc; i++)
  {
    if ((strcmp(argv[i], "-N") == 0) || (strcmp(argv[i], "-Rows") == 0))
    {
      N = pow(2, atoi(argv[++i]));
      printf("  User N is %ld\n", N);
    }
    else if ((strcmp(argv[i], "-M") == 0) || (strcmp(argv[i], "-Columns") == 0))
    {
      M = pow(2, atof(argv[++i]));
      printf("  User M is %ld\n", M);
    }
    else if ((strcmp(argv[i], "-S") == 0) || (strcmp(argv[i], "-Size") == 0))
    {
      S = pow(2, atof(argv[++i]));
      printf("  User S is %ld\n", S);
    }
    else if (strcmp(argv[i], "-threads") == 0)
    {
      num_threads = atoi(argv[++i]);
    }
    else if ((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "-help") == 0))
    {
      printf("  y^T*A*x Options:\n");
      printf("  -Rows (-N) <int>:      exponent num, determines number of rows 2^num (default: 2^12 = 4096)\n");
      printf("  -Columns (-M) <int>:   exponent num, determines number of columns 2^num (default: 2^10 = 1024)\n");
      printf("  -Size (-S) <int>:      exponent num, determines total matrix size 2^num (default: 2^22 = 4096*1024 )\n");
      printf("  -help (-h):            print this message\n\n");
      exit(1);
    }
  }

  // Check sizes.
  checkSizes(N, M, S);

  int batch = M / num_threads;
  size_t block_mem_size = sizeof(float) * num_threads;

  float *partial_result, *d_partial_result;

  float *x, *y, *A, *d_x, *d_y, *d_A; 

  A = (float*)malloc(sizeof(float) * N*M);
  x = (float*)malloc(sizeof(float) * M);
  y = (float*)malloc(sizeof(float) * N);

  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      A[m*N+n] = 1;
    }
  } 

  for (int m = 0; m < M; m++) {
    x[m] = 1;
  } 

  for (int n = 0; n < N; n++) {
    y[n] = 1;
  }

  struct timeval begin, end;

  gettimeofday(&begin, NULL);

  // Allocate host memory
  partial_result = (float*)malloc(sizeof(float) * N);

  // Allocate device memory
  cudaMalloc((void**)&d_partial_result, sizeof(float) * N);
  cudaMalloc((void**)&d_x, sizeof(float) * M);
  cudaMalloc((void**)&d_y, sizeof(float) * N);
  cudaMalloc((void**)&d_A, sizeof(float) * M*N);

  // Transfer data to the device memory
  cudaMemcpy(d_x, x, sizeof(float) * M, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_A, A, sizeof(float) * M*N, cudaMemcpyHostToDevice);

  float result = 0.0;

  calculate<<<N, num_threads, block_mem_size>>>(d_partial_result, d_A, d_x, d_y, M, N, batch);

  cudaDeviceSynchronize();

  // Transfer data back to host memory
  cudaMemcpy(partial_result, d_partial_result, sizeof(float) * N, cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++) {
    result += partial_result[i];
  }

  const float solution = (float) N * (float) M;

  if (result != solution)
  {
    printf("  Error: result( %lf ) != solution( %lf )\n", result, solution);
  }

  // Deallocate device memory
  cudaFree(d_partial_result);

  // Deallocate host memory
  free(partial_result);
  free(x);
  free(y);
  free(A);

  gettimeofday(&end, NULL);

  // Calculate time.
  // double time = timer.seconds();
  double time = 1.0 * (end.tv_sec - begin.tv_sec) +
                1.0e-6 * (end.tv_usec - begin.tv_usec);

  // Calculate bandwidth.
  // Each matrix A row (each of length M) is read once.
  // The x vector (of length M) is read N times.
  // The y vector (of length N) is read once.
  // double Gbytes = 1.0e-9 * double( sizeof(double) * ( 2 * M * N + N ) );
  double Gbytes = 1.0e-9 * double(sizeof(double) * (M + M * N + N));

  std::ofstream file;
  file.open("vector_shared.csv", std::ios_base::app);

  // Print results (problem size, time and bandwidth in GB/s).
  // printf("  N( %d ) M( %d ) nrepeat ( %d ) problem( %g MB ) time( %g s ) bandwidth( %g GB/s )\n",
  //        N, M, nrepeat, Gbytes * 1000, time, Gbytes * nrepeat / time);

  file << num_threads << "," << N << "," << M << "," << Gbytes << "," << time << std::endl;

  file.close();

  return 0;
}

void checkSizes(long &N, long &M, long &S)
{
  // If S is undefined and N or M is undefined, set S to 2^22 or the bigger of N and M.
  if (S == -1 && (N == -1 || M == -1))
  {
    S = pow(2, 22);
    if (S < N)
      S = N;
    if (S < M)
      S = M;
  }

  // If S is undefined and both N and M are defined, set S = N * M.
  if (S == -1)
    S = N * M;

  // If both N and M are undefined, fix row length to the smaller of S and 2^10 = 1024.
  if (N == -1 && M == -1)
  {
    if (S > 1024)
    {
      M = 1024;
    }
    else
    {
      M = S;
    }
  }

  // If only M is undefined, set it.
  if (M == -1)
    M = S / N;

  // If N is undefined, set it.
  if (N == -1)
    N = S / M;

  printf("  Total size S = %ld N = %ld M = %ld\n", S, N, M);

  // Check sizes.
  if ((S < 0) || (N < 0) || (M < 0))
  {
    printf("  Sizes must be greater than 0.\n");
    exit(1);
  }

  if ((N * M) != S)
  {
    printf("  N * M != S\n");
    exit(1);
  }
}
