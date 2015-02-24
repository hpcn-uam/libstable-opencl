#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include "stable_api.h"
#include "benchmarking.h"
#include "opencl_integ.h"

int main(int argc, char **argv)
{
	double alfa = 0.9;
    double beta = 0.91;
    int param = 0;
	double sigma = 1.0, mu = 0.0;
	int min_x_range = -5;
	int max_x_range = 5;
	int num_samples = 10000;
	double *x;
	double *pdf, *cpu_pdf;
	double *errs, *cpu_errs;
	double x_step_size = ((double)(max_x_range - min_x_range)) / (double) num_samples;
	StableDist* dist;
	double abserr = 0, relerr = 0, cpu_err = 0, gpu_err = 0;
	double abserr_v = 0, relerr_v = 0, cpu_err_v = 0, gpu_err_v = 0;
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
		abserr_v += (fabs(pdf[i] - cpu_pdf[i])) * (fabs(pdf[i] - cpu_pdf[i]));
		relerr += fabs(pdf[i] - cpu_pdf[i]) / cpu_pdf[i];
		relerr_v += (fabs(pdf[i] - cpu_pdf[i]) / cpu_pdf[i]) * (fabs(pdf[i] - cpu_pdf[i]) / cpu_pdf[i]);
		cpu_err += cpu_errs[i];
		cpu_err_v += (cpu_errs[i]) * (cpu_errs[i]);
		gpu_err += errs[i];
		gpu_err_v += (errs[i]) * (errs[i]);
		printf("%lf %lf %lf %lf %lf\n", x[i], pdf[i], cpu_pdf[i], errs[i], cpu_errs[i]);
	}

	abserr /= num_samples;
	abserr_v = sqrt((abserr_v / num_samples - abserr * abserr) * num_samples / (num_samples - 1));
	relerr /= num_samples;
	relerr_v = sqrt((relerr_v / num_samples - relerr * relerr) * num_samples / (num_samples - 1));
	cpu_err /= num_samples;
	cpu_err_v = sqrt((cpu_err_v / num_samples - cpu_err * cpu_err) * num_samples / (num_samples - 1));
	gpu_err /= num_samples;
	gpu_err_v = sqrt((gpu_err_v / num_samples - gpu_err * gpu_err) * num_samples / (num_samples - 1));

	fprintf(stderr, "Average absolute error: %g ± %g\n", abserr, abserr_v);
	fprintf(stderr, "Average relative error: %g ± %g\n", relerr, relerr_v);
	fprintf(stderr, "Average cpu error: %g ± %g\n", cpu_err, cpu_err_v);
	fprintf(stderr, "Average gpu error: %g ± %g\n", gpu_err, gpu_err_v);
	fprintf(stderr, "ξ is %g\n", - beta * tan(alfa * M_PI_2));

	stable_free(dist);
	return 0;
}
