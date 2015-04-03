/* tests/fittest
 *
 * Example program that test Libstable parameter estimation methods.
 *
 * Copyright (C) 2013. Javier Royuela del Val
 *                     Federico Simmross Wattenberg
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; If not, see <http://www.gnu.org/licenses/>.
 *
 *
 *  Javier Royuela del Val.
 *  E.T.S.I. Telecomunicación
 *  Universidad de Valladolid
 *  Paseo de Belén 15, 47002 Valladolid, Spain.
 *  jroyval@lpi.tel.uva.es
 */
#include "stable_api.h"
#include "benchmarking.h"
#include <time.h>
#include <stdlib.h>

typedef int (*fitter)(StableDist *, const double *, const unsigned int);

struct fittest
{
	fitter func;
	short gpu_enabled;
	const char *name;
};

struct fitresult
{
	double ms_duration;
	double alfa;
	double beta;
	double sigma;
	double mu_0;
	double alfa_err;
	double beta_err;
	double sigma_err;
	double mu_0_err;
};

#define calc_avg_err(variable) do { \
	result->variable /= Nexp; \
	result->variable ## _err = sqrt((result->variable ## _err / Nexp - result->variable * result->variable) * Nexp / (Nexp - 1)); \
} while (0)

#define add_avg_err(variable) do { \
	result->variable += dist->variable; \
	result->variable ## _err += dist->variable * dist->variable; \
} while(0)

#define print_deviation(variable) do { \
	double dev = fabs(result->variable - variable); \
	double perc_dev; \
	if (variable == 0) \
		perc_dev = 100 * dev; \
	else \
		perc_dev = 100 * dev / variable; \
	acc_pdev += perc_dev; \
	printf("\t%.3lf (%.1lf %%)", dev, perc_dev); \
} while(0)

int main (int argc, char *argv[])
{
	double alfa, beta, sigma, mu_0;
	double *data;
	int i = 1, iexp, N, Nexp;
	int seed;
	double acc_pdev;
	double total_duration, start, end;
	struct fittest tests[] =
	{
		{ stable_fit_mle, 0, "MLE" },
		{ stable_fit_mle2d, 0, "M2D"},
		{ stable_fit_koutrouvelis, 0, "KTR"},
		{ stable_fit_mle, 1, "MLE" },
		{ stable_fit_mle2d, 1, "M2D"},
		{ stable_fit_koutrouvelis, 1, "KTR"},
	};
	struct fittest *test;
	size_t num_tests = sizeof tests / sizeof(struct fittest);
	struct fitresult* results;
	struct fitresult* result;

	results = calloc(num_tests, sizeof(struct fitresult));

	StableDist *dist = NULL;

	alfa = 1.5;
	beta = 0.75;
	sigma = 5.0;
	mu_0 = 15.0;
	N = 400;
	Nexp = 5;
	seed = -1;

	printf("Parameters for the random data generated:\n");
	printf("α\t%lf\n", alfa);
	printf("β\t%lf\n", beta);
	printf("σ\t%lf\n", sigma);
	printf("μ\t%lf\n", mu_0);
	printf("Size\t%d\n", N);
	printf("\nWill perform %d experiments for each fitter.\n\n", Nexp);

	if ((dist = stable_create(alfa, beta, sigma, mu_0, 0)) == NULL)
	{
		printf("Error when creating the distribution");
		exit(1);
	}

	stable_set_THREADS(1);
	stable_set_absTOL(1e-16);
	stable_set_relTOL(1e-8);
	stable_set_FLOG("errlog.txt");

	if (seed < 0)
		stable_rnd_seed(dist, time(NULL));
	else
		stable_rnd_seed(dist, seed);

	/* Random sample generation */
	data = (double *)malloc(N * Nexp * sizeof(double));

	stable_rnd(dist, data, N * Nexp);

	printf("Fitter\tms/fit\t\tα\t\tβ\t\tμ\t\tσ\n");

	for (i = 0; i < num_tests; i++)
	{
		test = tests + i;
		result = results + i;
		total_duration = 0;

		for (iexp = 0; iexp < Nexp; iexp++)
		{
			stable_fit_init(dist, data + iexp * N, N, NULL, NULL);

			if (test->gpu_enabled)
				stable_activate_gpu(dist);
			else
				stable_deactivate_gpu(dist);

			start = get_ms_time();
			test->func(dist, data + iexp * N, N);
			end = get_ms_time();

			add_avg_err(alfa);
			add_avg_err(beta);
			add_avg_err(sigma);
			add_avg_err(mu_0);

			result->ms_duration += end - start;
		}

		calc_avg_err(alfa);
		calc_avg_err(beta);
		calc_avg_err(sigma);
		calc_avg_err(mu_0);
		result->ms_duration /= Nexp;

		printf("%s", test->name);

		if (test->gpu_enabled)
			printf("_GPU");

		printf("\t%lf\t%.2lf ± %.2lf\t%.2lf ± %.2lf\t%.2lf ± %.2lf\t%.2lf ± %.2lf\n",
		       result->ms_duration,
		       result->alfa, result->alfa_err,
		       result->beta, result->beta_err,
		       result->sigma, result->sigma_err,
		       result->mu_0, result->mu_0_err);
	}

	printf("\n\nComparison of actual vs. expected results:\n");
	printf("Fitter\tα error\t\tβ error\t\tμ error\t\tσ error\t\tAverage %% error\n");

	for(i = 0; i < num_tests; i++)
	{
		result = results + i;
		test = tests + i;
		acc_pdev = 0;

		printf("%s", test->name);

		if (test->gpu_enabled)
			printf("_GPU");

		print_deviation(alfa);
		print_deviation(beta);
		print_deviation(mu_0);
		print_deviation(sigma);

		printf("\t%.1lf %%\n", acc_pdev / 4);
	}


	free(data);
	free(results);
	stable_free(dist);

	fclose(stable_get_FLOG());

	return 0;
}
