#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include "stable_api.h"
#include "opencl_integ.h"
#include "methods.h"

int main(int argc, char **argv)
{
	size_t args_per_component = 5;
	StableDist* dist;
	size_t i;
	double xmin, xmax;
	double resolution = 0.001;
	double *x, *pdf, step;
	size_t extra_args = 2;
	size_t num_samples;

	dist = stable_create(1, 0, 1, 0, 0);

	stable_set_mixture_components(dist, (argc - 1 - extra_args) / args_per_component);

	xmin = strtod(argv[1], NULL);
	xmax = strtod(argv[2], NULL);

	num_samples = (xmax - xmin) / resolution;
	pdf = calloc(num_samples, sizeof(double));

	for (i = 0; i < dist->num_mixture_components; i++) {
		dist->mixture_components[i]->alfa = strtod(argv[1 + extra_args + i * args_per_component], NULL);
		dist->mixture_components[i]->beta = strtod(argv[2 + extra_args + i * args_per_component], NULL);
		dist->mixture_components[i]->mu_0 = strtod(argv[3 + extra_args + i * args_per_component], NULL);
		dist->mixture_components[i]->sigma = strtod(argv[4 + extra_args + i * args_per_component], NULL);
		dist->mixture_weights[i] = strtod(argv[5 + extra_args + i * args_per_component], NULL);
	}

	vector_npoints(&x, xmin, xmax, num_samples, &step);
	stable_pdf(dist, x, num_samples, pdf, NULL);

	for (i = 0; i < num_samples; i++)
		printf("%lf 0 %lf\n", x[i], pdf[i]);

	stable_free(dist);
	return 0;
}
