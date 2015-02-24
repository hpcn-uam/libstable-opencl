#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include "stable_api.h"
#include "benchmarking.h"
#include "opencl_integ.h"

int main(int argc, char **argv)
{
	double alfa = 1.25;
    double beta = 0.5;
    int param = 0;
	double sigma = 1.0, mu = 0.0;
	int min_x_range = -3;
	int max_x_range = 3;
	int num_samples = 4000;
	double *x;
	double *pdf, *cpu_pdf;
	double *errs, *cpu_errs;
	double x_step_size = ((double)(max_x_range - min_x_range)) / (double) num_samples;
	StableDist* dist;
	double abserr = 0, relerr = 0, cpu_err = 0, gpu_err = 0;
	int i;

	if (argc == 3)
	{
		alfa = strtod(argv[1], NULL);
		beta = strtod(argv[2], NULL);
	}

	dist = stable_create(alfa, beta, sigma, mu, param);
	x = calloc(num_samples, sizeof(double));
	pdf = calloc(num_samples, sizeof(double));
	cpu_pdf = calloc(num_samples, sizeof(double));
	errs = calloc(num_samples, sizeof(double));
	cpu_errs = calloc(num_samples, sizeof(double));

	if (!dist)
	{
		fprintf(stderr, "StableDist creation failure. Aborting.\n");
		return 1;
	}

	for (i = 0; i < num_samples; i++)
		x[i] = min_x_range + x_step_size * i;

	if (stable_activate_gpu(dist))
	{
		fprintf(stderr, "Couldn't initialize GPU.\n");
		return 1;
	}


	stable_pdf(dist, x, num_samples, cpu_pdf, errs);
	stable_pdf_gpu(dist, x, num_samples, pdf, cpu_errs);

	for (i = 0; i < num_samples; i++)
	{
		abserr += fabs(pdf[i] - cpu_pdf[i]);
		relerr += fabs(pdf[i] - cpu_pdf[i]) / cpu_pdf[i];
		cpu_err += cpu_errs[i];
		gpu_err += errs[i];
		printf("%lf %lf %lf %lf %lf\n", x[i], pdf[i], cpu_pdf[i], errs[i], cpu_errs[i]);
	}

	fprintf(stderr, "Average absolute error: %g\n", abserr / num_samples);
	fprintf(stderr, "Average relative error: %g\n", relerr / num_samples);
	fprintf(stderr, "Average cpu error: %g\n", cpu_err / num_samples);
	fprintf(stderr, "Average gpu error: %g\n", gpu_err / num_samples);

	stable_free(dist);
	return 0;
}
