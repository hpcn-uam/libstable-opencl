
double kstest(const double* samples, size_t nsamples, const double* cdf, double* d)
{
	double max_diff = 0;

	for (size_t i = 0; i < nsamples; i++) {
		double ecdf = ((double) i) / nsamples;
		double diff = fabs(ecdf - cdf[i]);

		if (max_diff < diff)
			max_diff = diff;
	}

	if (d != NULL)
		*d = max_diff;

	double sqrt_n = sqrt(((double) nsamples));
	return probks((sqrt_n + 0.12 + 0.11 / sqrt_n) * max_diff);
}

double probks(double alam)
{
	// Source: Numerical recipes in C: The art of scientific computing.
	int j;
	double a2, fac = 2.0, sum = 0.0, term, termbf = 0.0;
	a2 = -2.0 * alam * alam;

	for (j = 1; j <= 100; j++) {
		term = fac * exp(a2 * j * j);
		sum += term;

		if (fabs(term) <= 0.001 * termbf || fabs(term) <= 1.0e-8 * sum) return sum;

		fac = -fac;
		termbf = fabs(term);
	}

	return 1.0;
}

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
