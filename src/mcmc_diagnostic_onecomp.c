/*
 * Copyright (C) 2015 - Naudit High Performance Computing and Networking
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
 */

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <assert.h>
#include "stable_api.h"
#include "benchmarking.h"
#include "opencl_integ.h"
#include "kde.h"
#include "methods.h"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#define MAX_POINTS 50000

volatile sig_atomic_t stop_prog = 0;
extern volatile sig_atomic_t stop;

void _handle_signal(int sig)
{
	printf("Stopping...\n");
	stop_prog = 1;
	stop = 1;
}

int main(int argc, char **argv)
{
	size_t num_points = 300;
	size_t num_components = 1;
	size_t param_idx;
	size_t i, j;
	double alpha = 0.5, beta = 0, mu = 0, sigma = 1;
	double min_alpha = 0.05, min_beta = 0.95;
	double max_alpha = 1.95, max_beta = 0.95;
	double step = 0.1;
	size_t num_chains = 10;
	size_t chain;
	size_t lag;
	size_t min_lag = 0;
	double autocorr;
	struct stable_mcmc_settings settings[num_chains];
	double correlations[MAX_STABLE_PARAMS][MAX_STABLE_PARAMS];
	double orig_params[MAX_STABLE_PARAMS], new_params[MAX_STABLE_PARAMS];
	double deviations[MAX_STABLE_PARAMS];
	size_t min_lags[MAX_STABLE_PARAMS];

	assert(MAX_POINTS >= num_points);

	double rnd[num_points];
	FILE* outfile;
	int retval = EXIT_SUCCESS;

	StableDist* dist;

	gsl_set_error_handler_off();

	dist = stable_create(alpha, beta, sigma, mu, 0);

	if (!dist) {
		fprintf(stderr, "StableDist creation failure. Aborting.\n");
		return 1;
	}

	outfile = fopen("mcmc_diag.dat", "w");

	if (!outfile) {
		perror("fopen");
		retval = EXIT_FAILURE;
		goto out;
	}

	signal(SIGINT, _handle_signal);
	signal(SIGTERM, _handle_signal);

	for (alpha = min_alpha; alpha <= max_alpha && !stop_prog; alpha += step) {
		for (beta = min_beta; beta <= max_beta && !stop_prog; beta += step) {
			printf("Generating %zu random numbers for α = %.2lf, β = %.2lf...\n", num_points, alpha, beta);
			stable_set_mixture_components(dist, 1);
			dist->mixture_weights[0] = 1.0;
			stable_setparams(dist->mixture_components[0], alpha, beta, sigma, mu, 0);

			stable_getparams_array(dist->mixture_components[0], orig_params);

			stable_rnd(dist, rnd, num_points);
			printf("Starting mixture estimation.\n");
			min_lag = 0;

			for (i = 0; i < MAX_STABLE_PARAMS; i++) {
				deviations[i] = 0;

				for (j = 0; j < MAX_STABLE_PARAMS; j++)
					correlations[i][j] = 0;
			}

			for (chain = 0; chain < num_chains && !stop_prog; chain++) {
				do {
					for (param_idx = 0; param_idx < MAX_STABLE_PARAMS; param_idx++)
						new_params[param_idx] = gsl_ran_gaussian(dist->gslrand, 0.5) + orig_params[param_idx];
				} while (stable_checkparams(new_params[0], new_params[1], new_params[2], new_params[3], 1) == NOVALID);

				printf("Estimating chain %zu/%zu...\n", chain, num_chains);

				stable_fit_mixture_default_settings(settings + chain);
				settings[chain].skip_initial_estimation = 1;
				settings[chain].fix_components = 1;
				settings[chain].location_lock_iterations = 0;
				settings[chain].fix_components_during_last_n_iterations = 0;
				settings[chain].max_iterations = 1000;
				settings[chain].handle_signal = 0;
				sprintf(settings[chain].debug_data_fname, "mcmc_diag/mixture_debug_a%.2lf_b%.2lf_chain%zu.dat", alpha, beta, chain);

				stable_fit_mixture_settings(dist, rnd, num_points, settings + chain);

				for (i = 0; i < MAX_STABLE_PARAMS; i++) {
					deviations[i] += fabs(settings[chain].final_param_avg[0][i] - orig_params[i]) / num_chains;

					for (j = 0; j < MAX_STABLE_PARAMS; j++)
						correlations[i][j] += settings[chain].correlations[0][i][j] / num_chains;

					autocorr = 20;

					for (lag = 1; lag < 50 && fabs(autocorr) > 0.15; lag++)
						autocorr = autocorrelation(settings[chain].param_values[0][i], settings[chain].num_iterations, lag);

					if (lag > min_lag)
						min_lag = lag;
				}

				stable_fit_mixture_print_results(settings + chain);
			}

			for (i = 0; i < MAX_STABLE_PARAMS; i++)
				printf("PSRF %s: %.3lf\n", _param_names[i], gelman_rubin(settings, num_chains, i));

			printf("Thinning: %zu\n", min_lag);
		}
	}

	printf("Done\n");

out:
	stable_free(dist);

	return retval;
}
