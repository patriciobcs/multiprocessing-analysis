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

double critical()
{
  double x, sum = 0.0;
#pragma omp parallel for shared(sum) private(x) num_threads(num_threads)
  for (int i = 1; i <= num_steps; i++)
  {
    x = (i - 0.5) * step;
    double add = 4.0 / (1.0 + x * x);
#pragma omp critical
    sum += add;
  }
  return step * sum;
}

double atomic()
{
  double x, sum = 0.0;
#pragma omp parallel for shared(sum) private(x) num_threads(num_threads)
  for (int i = 1; i <= num_steps; i++)
  {
    x = (i - 0.5) * step;
    double add = 4.0 / (1.0 + x * x);
#pragma omp atomic
    sum += add;
  }
  return step * sum;
}

double reduction()
{
  double x, sum = 0.0;
#pragma omp parallel for reduction(+ \
                                   : sum) private(x) num_threads(num_threads)
  for (int i = 1; i <= num_steps; i++)
  {
    x = (i - 0.5) * step;
    double add = 4.0 / (1.0 + x * x);
    sum += add;
  }
  return step * sum;
}

double split()
{
  const int N = 100;
  const int slice_size = num_steps / N;
  double x, sum = 0.0;
#pragma omp parallel for reduction(+ \
                                   : sum) private(x) num_threads(num_threads)
  for (int j = 0; j < N; j++)
  {
    double partial_sum = 0.0;
    int start = 1 + j * slice_size;
    int end = start + slice_size;

    for (int i = start; i <= end; i++)
    {
      x = (i - 0.5) * step;
      partial_sum += 4.0 / (1.0 + x * x);
    }
    sum += partial_sum;
  }
  return step * sum;
}

int main(int argc, char **argv)
{
  bool run_critical, run_atomic, run_reduction, run_split = false;

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
    else if ((strcmp(argv[i], "-critical") == 0))
    {
      run_critical = true;
    }
    else if ((strcmp(argv[i], "-atomic") == 0))
    {
      run_atomic = true;
    }
    else if ((strcmp(argv[i], "-reduction") == 0))
    {
      run_reduction = true;
    }
    else if ((strcmp(argv[i], "-split") == 0))
    {
      run_split = true;
    }
  }

  std::ofstream file;
  file.open("metrics_part_1.csv", std::ios_base::app);

  double pi, time = 0.0;

  step = 1.0 / (double)num_steps;

  struct timeval begin, end;

  if (run_critical)
  {
    // Critical
    gettimeofday(&begin, NULL);

    pi = critical();

    gettimeofday(&end, NULL);
    time = 1.0 * (end.tv_sec - begin.tv_sec) +
           1.0e-6 * (end.tv_usec - begin.tv_usec);
    file << "critical"
         << "," << num_steps << "," << num_threads << "," << pi << "," << time << std ::endl;
  }

  if (run_atomic)
  {
    // Atomic
    gettimeofday(&begin, NULL);

    pi = atomic();

    gettimeofday(&end, NULL);
    time = 1.0 * (end.tv_sec - begin.tv_sec) +
           1.0e-6 * (end.tv_usec - begin.tv_usec);
    file << "atomic"
         << "," << num_steps << "," << num_threads << "," << pi << "," << time << std ::endl;
  }

  if (run_reduction)
  {
    // Reduction
    gettimeofday(&begin, NULL);

    pi = reduction();

    gettimeofday(&end, NULL);
    time = 1.0 * (end.tv_sec - begin.tv_sec) +
           1.0e-6 * (end.tv_usec - begin.tv_usec);
    file << "reduction"
         << "," << num_steps << "," << num_threads << "," << pi << "," << time << std ::endl;
  }

  if (run_split)
  {
    // Split
    gettimeofday(&begin, NULL);

    pi = split();

    gettimeofday(&end, NULL);
    time = 1.0 * (end.tv_sec - begin.tv_sec) +
           1.0e-6 * (end.tv_usec - begin.tv_usec);
    file << "split"
         << "," << num_steps << "," << num_threads << "," << pi << "," << time << std ::endl;
  }

  file.close();
}