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
	size_t extra_args = 3;
	size_t num_samples;
	double samples[10000];
	double ks, hl, kl;
	FILE* fsamples;

	dist = stable_create(1, 0, 1, 0, 0);

	stable_set_mixture_components(dist, (argc - 1 - extra_args) / args_per_component);

	stable_set_relTOL(1e-2);
	stable_set_absTOL(1e-5);

	xmin = strtod(argv[2], NULL);
	xmax = strtod(argv[3], NULL);

	fsamples = fopen(argv[1], "r");

	for (i = 0; fscanf(fsamples, "%lf", samples + i) == 1 && i < 10000; i++);

	num_samples = i;

	for (i = 0; i < dist->num_mixture_components; i++) {
		stable_setparams(dist->mixture_components[i],
						 strtod(argv[1 + extra_args + i * args_per_component], NULL),
						 strtod(argv[2 + extra_args + i * args_per_component], NULL),
						 strtod(argv[4 + extra_args + i * args_per_component], NULL),
						 strtod(argv[3 + extra_args + i * args_per_component], NULL),
						 0);
		dist->mixture_weights[i] = strtod(argv[5 + extra_args + i * args_per_component], NULL);
	}

	stable_kolmogorov_smirnov_gof(dist, samples, num_samples, &ks);
	hl = stable_hellinger_gof(dist, samples, num_samples);
	kl = stable_kullback_leibler_gof(dist, samples, num_samples);

	printf("KS: %lf\nHellinger: %lf\nKullbackLeibler: %lf\n", ks, hl, kl);

	stable_free(dist);
	return 0;
}
