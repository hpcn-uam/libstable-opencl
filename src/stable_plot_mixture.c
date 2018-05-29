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
	size_t i, j;
	double xmin, xmax;
	double resolution = 0.001;
	double *x, *pdf, step, **pdf_comp;
	size_t extra_args = 2;
	size_t num_samples;

	dist = stable_create(1, 0, 1, 0, 0);

	stable_set_mixture_components(dist, (argc - 1 - extra_args) / args_per_component);

	stable_set_relTOL(1e-2);
	stable_set_absTOL(1e-5);

	xmin = strtod(argv[1], NULL);
	xmax = strtod(argv[2], NULL);

	num_samples = 1000;
	pdf = calloc(num_samples, sizeof(double));
	pdf_comp = calloc(dist->num_mixture_components, sizeof(double));

	vector_npoints(&x, xmin, xmax, num_samples, &step);

	for (i = 0; i < dist->num_mixture_components; i++) {
		stable_setparams(dist->mixture_components[i],
						 strtod(argv[1 + extra_args + i * args_per_component], NULL),
						 strtod(argv[2 + extra_args + i * args_per_component], NULL),
						 strtod(argv[4 + extra_args + i * args_per_component], NULL),
						 strtod(argv[3 + extra_args + i * args_per_component], NULL),
						 0);
		dist->mixture_weights[i] = strtod(argv[5 + extra_args + i * args_per_component], NULL);

		pdf_comp[i] = calloc(num_samples, sizeof(double));
		stable_pdf(dist->mixture_components[i], x, num_samples, pdf_comp[i], NULL);
	}

	fprintf(stderr, "%zu components\n", dist->num_mixture_components);

	stable_pdf(dist, x, num_samples, pdf, NULL);

	for (i = 0; i < num_samples; i++) {
		printf("%lf 0 %lf", x[i], pdf[i]);

		for (j = 0; j < dist->num_mixture_components; j++)
			printf(" %lf", pdf_comp[j][i] * dist->mixture_weights[j]);

		printf("\n");
	}


	stable_free(dist);
	return 0;
}
