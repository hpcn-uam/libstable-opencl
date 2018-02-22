#include <gsl/gsl_sort.h>

#include "stable_api.h"
#include "methods.h"

double stable_kolmogorov_smirnov_gof(StableDist* dist, const double* samples, size_t nsamples)
{
	double *cdf, *samples_sorted;
	double d;
	double result;

	cdf = calloc(nsamples, sizeof(double));
	samples_sorted = calloc(nsamples, sizeof(double));

	memcpy(samples_sorted, samples, nsamples * sizeof(double));
	gsl_sort(samples_sorted, 1, nsamples);

	if (dist->gpu_enabled)
		stable_cdf_gpu(dist, samples_sorted, nsamples, cdf, NULL);
	else
		stable_cdf(dist, samples_sorted, nsamples, cdf, NULL);

	result = kstest(samples_sorted, nsamples, cdf, &d);

	free(cdf);
	free(samples_sorted);

	return result;
}
