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
#include "stable_gridfit.h"
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
	variable ## _est /= Nexp; \
	variable ## _est_err = sqrt((variable ## _est_err / Nexp - variable ## _est * variable ## _est) * Nexp / (Nexp - 1)); \
} while (0)

#define add_avg_err(variable) do { \
	variable ## _est += dist->variable; \
	variable ## _est_err += dist->variable * dist->variable; \
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

#define add_initial_estimations(variable) do { \
	variable ## _init += dist->variable; \
} while(0)

#define ALFA_START 0.1
#define ALFA_END 1.9
#define ALPHA_INCR 0.05
#define BETA_START -0.9
#define BETA_END 0.9
#define BETA_INCR 0.05
#define MU_START -10
#define MU_END 10
#define MU_INCR 1
#define SIGMA_END 30
#define SIGMA_INCR 0.5
#define SIGMA_START SIGMA_INCR

int main (int argc, char *argv[])
{
	double alfa, beta, sigma, mu_0;
	double *data;
	int i = 1, iexp, N, Nexp;
	int seed;
	double total_duration, start, end;
	struct fittest tests[] =
	{
		//{ stable_fit_mle, 0, "MLE" },
		{ stable_fit_mle2d, 0, "M2D"},
		{ stable_fit_koutrouvelis, 0, "KTR"},
		{ stable_fit_mle, 1, "MLE" },
		{ stable_fit_mle2d, 1, "M2D"},
		{ stable_fit_koutrouvelis, 1, "KTR"},
		{ stable_fit_grid, 1, "GRD" },
		{ stable_fit_grid, 0, "GRD" }
	};
	struct fittest *test;
	size_t num_tests = sizeof tests / sizeof(struct fittest);

	StableDist *dist = NULL;

	alfa = 1.5;
	beta = 0.75;
	sigma = 5.0;
	mu_0 = 15.0;
	N = 400;
	Nexp = 20;
	seed = -1;

	if ((dist = stable_create(alfa, beta, sigma, mu_0, 0)) == NULL)
	{
		printf("Error when creating the distribution");
		exit(1);
	}

	stable_set_THREADS(1);
	stable_set_absTOL(1e-16);
	stable_set_relTOL(1e-8);

	if (seed < 0)
		stable_rnd_seed(dist, time(NULL));
	else
		stable_rnd_seed(dist, seed);

	/* Random sample generation */
	data = (double *) malloc(Nexp * sizeof(double));

	stable_rnd(dist, data, Nexp);


	for (i = 0; i < num_tests; i++)
	{
		test = tests + i;
		total_duration = 0;

		char out_fname[100];
		char* gpu_marker;
		if(test->gpu_enabled)
			gpu_marker = "_GPU";
		else
			gpu_marker = "";


		snprintf(out_fname, 100, "%s%s.dat", test->name, gpu_marker);

		FILE* out = fopen(out_fname, "w");

		if(!out)
		{
			perror("fopen");
			return 1;
		}

		printf("Estimation evaluation for %s%s...\n", test->name, gpu_marker);

		for(alfa = ALFA_START; alfa <= ALFA_END; alfa += ALPHA_INCR)
		{
			for(beta = BETA_START; beta <= BETA_END; beta += BETA_INCR)
			{
				mu_0 = 0;
				sigma = 1;
				//for(mu_0 = MU_START; mu_0 <= MU_END; mu_0 += MU_INCR)
				{
				//	for(sigma = SIGMA_START; sigma <= SIGMA_END; sigma += SIGMA_INCR)
					{
						double alfa_est = 0, beta_est = 0, mu_0_est = 0, sigma_est = 0;
						double alfa_est_err = 0, beta_est_err = 0, mu_0_est_err = 0, sigma_est_err = 0;
						stable_setparams(dist, alfa, beta, sigma, mu_0, 0);
						stable_rnd(dist, data, Nexp);
						double ms_duration = 0;

						if (test->gpu_enabled)
							stable_activate_gpu(dist);
						else
							stable_deactivate_gpu(dist);

						dist->parallel_gridfit = test->gpu_enabled; // Temporary.

						for (iexp = 0; iexp < Nexp; iexp++)
						{
							if(stable_fit_init(dist, data + iexp * N, N, NULL, NULL) != 0)
								continue;

							start = get_ms_time();
							test->func(dist, data + iexp * N, N);
							end = get_ms_time();

							add_avg_err(alfa);
							add_avg_err(beta);
							add_avg_err(sigma);
							add_avg_err(mu_0);

							ms_duration += end - start;
						}

						calc_avg_err(alfa);
						calc_avg_err(beta);
						calc_avg_err(sigma);
						calc_avg_err(mu_0);

						ms_duration /= Nexp;

						fprintf(out, "%lf %lf %lf %lf %lf ", alfa, beta, mu_0, sigma, ms_duration);
						fprintf(out, "%lf %lf %lf %lf %lf %lf %lf %lf\n",
					       alfa_est, alfa_est_err,
					       beta_est, beta_est_err,
					       sigma_est, sigma_est_err,
					       mu_0_est, mu_0_est_err);
					}

				}
			}
		}

		fclose(out);
		printf("Eval finished.\n");
	}

	printf("Done\n");


	free(data);
	stable_free(dist);

	fclose(stable_get_FLOG());

	return 0;
}
