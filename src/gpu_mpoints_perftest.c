#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include "stable_api.h"
#include "benchmarking.h"
#include "opencl_integ.h"

int main (void)
{
	double alfa = 1.25, beta = 0.5, sigma = 1.0, mu = 0.0;
	int param = 0;
	int max_test_size = 20000;
	int num_tests_per_size = 4;
	int test_size_step = 10;
	int test_size;
	int i;
	double *x;
	double *pdf;
	StableDist *dist;
	int min_x_range = -20;
	int max_x_range = -min_x_range;
	double x_step_size = ((double)(max_x_range - min_x_range)) / (double) max_test_size;
	double start, end, duration;

	dist = stable_create(alfa, beta, sigma, mu, param);
	x = calloc(max_test_size, sizeof(double));
	pdf = calloc(max_test_size, sizeof(double));

	if (!dist)
	{
		fprintf(stderr, "StableDist creation failure. Aborting.\n");
		return 1;
	}

	for (i = 0; i < max_test_size; i++)
		x[i] = min_x_range + x_step_size * i;

	if (stable_activate_gpu(dist))
	{
		fprintf(stderr, "Couldn't initialize GPU.\n");
		return 1;
	}

	for (test_size = test_size_step; test_size <= max_test_size; test_size += test_size_step)
	{
		duration = 0;

		for (i = 0; i < num_tests_per_size; i++)
		{
			start = get_ms_time();
			stable_pdf_gpu(dist, x, test_size, pdf, NULL);
			end = get_ms_time();
			duration += end - start;
		}

        printf("%d\t%.3lf\t%.3lf\n", test_size, duration, duration / test_size);
	}

	stable_free(dist);
	return 0;
}
