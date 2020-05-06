/**
 * @author Guillermo Julián Moreno
 * @brief  This code creates a binary for mixture estimation.
 *
 * Usage of the binary: bin/release/mixtures [filename]
 *
 * where filename is an optional file with one value per line. The program will create
 * an estimation for the distribution of those values.
 *
 * If no filename is provided, the program will generate points from a known
 * alpha-stable mixture and try to estimate them.
 *
 * Copyright (C) 2018 - Naudit High Performance Computing and Networking
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
#include <time.h>
#include "stable_api.h"
#include "benchmarking.h"
#include "opencl_integ.h"
#include "kde.h"

#define MAX_POINTS 50000

#define SYN_CASE_2

int main(int argc, char **argv)
{
	size_t num_points = 1000;
	double epdf_resolution = 0.001;
	size_t min_points = 1000;
	size_t epdf_points;
	size_t i;
	struct stable_mcmc_settings settings;

	assert(MAX_POINTS >= num_points);

	/** Parameter definitions for synthetic cases, for tests */
#ifdef SYN_CASE_1
	double alphas[] = { 1.2, 0.8, 1.8 };
	double betas[] = { -0.5, 0.8, 0 };
	double mus[] = { -2, 0, 2 };
	double sigmas[] = { 0.25, 0.8, 0.2 };
	double weights[] = { 0.4, 0.5, 0.1 };
#endif


#ifdef SYN_CASE_2
	double alphas[] = { 1.2, 0.8, 2, 0.6 };
	double betas[] = { -0.5, 0.5, 0, 0 };
	double mus[] = { -2, 0, 2, 2.5 };
	double sigmas[] = { 0.5, 0.8, 0.15, 0.15 };
	double weights[] = { 0.2, 0.3, 0.3, 0.2 };
#endif

#ifdef SYN_CASE_3
	double alphas[] = { 0.6, 2 };
	double betas[] = { 0, 0 };
	double mus[] = {0, 0};
	double sigmas[] = {0.2, 2};
	double weights[] = {0.35, 0.65};
#endif

#ifdef SYN_CASE_4
	double alphas[] = { 1.6, 2, 1.2 };
	double betas[] = { -0.5, 0, 0.2 };
	double mus[] = { -5, 4, 15};
	double sigmas[] = {3.25, 3.5, 6};
	double weights[] = {0.25, 0.25, 0.5};
#endif

	// Only used to get number of components when generating synthetic sets
	size_t num_components = sizeof weights / sizeof(double);

	double rnd[MAX_POINTS];
	double *pdf, *cdf;
	double *pdf_predicted;
	double *x;
	double mn, mx;
	short has_real_pdf = 0;
	struct timespec t_start, t_end;

	FILE* infile = NULL;
	FILE* outfile;
	FILE* summary;
	int retval = EXIT_SUCCESS;

	StableDist* dist;

	clock_gettime(CLOCK_MONOTONIC, &t_start);

	if (argc == 2) {
		infile = fopen(argv[1], "r");

		if (!infile) {
			perror("fopen");
			return EXIT_FAILURE;
		}
	}

	// Avoid weird aborts
	gsl_set_error_handler_off();

	// Create the distribution object.
	dist = stable_create(alphas[0], betas[0], sigmas[0], mus[0], 0);

	if (!dist) {
		fprintf(stderr, "StableDist creation failure. Aborting.\n");
		return 1;
	}

	// Ingest data
	if (infile == NULL) {
		printf("Generating random numbers (%zu components)...\n", num_components);
		stable_set_mixture_components(dist, num_components);

		for (i = 0; i < dist->num_mixture_components; i++) {
			dist->mixture_weights[i] = weights[i];
			stable_setparams(dist->mixture_components[i], alphas[i], betas[i], sigmas[i], mus[i], 0);
		}

		stable_rnd(dist, rnd, num_points);
		has_real_pdf = 1;
	} else {
		printf("Reading from file %s... ", argv[1]);

		for (i = 0; fscanf(infile, "%lf", rnd + i) == 1 && i < MAX_POINTS; i++);

		num_points = i;
		printf("%zu records\n", i);

		has_real_pdf = 1;
	}

	// Records need to be sorted for plotting
	printf("Sorting records... ");
	gsl_sort(rnd, 1, num_points);
	printf("done\n");

	// Generate the EPDF for plotting of auxiliar files
	mn = rnd[0];
	mx = rnd[num_points - 1];

	printf("Data full range is [%lf:%lf]\n", mn, mx);

	if (mx - mn > 100) {
		double midpoint;

		if (has_real_pdf)
			midpoint = gsl_stats_mean(mus, 1, num_components);
		else
			midpoint = (mn + mx) / 2;

		mx = midpoint + 50;
		mn = midpoint - 50;
		printf("Range is too big, reducing to [%lf:%lf] for initial plotting (does not affect mixtures)\n", mn, mx);
	}

	epdf_points = (mx - mn) / epdf_resolution;

	if (epdf_points < min_points)
		epdf_points = min_points;

	// Allocate memory for epdf plots
	x = calloc(epdf_points, sizeof(double));
	pdf = calloc(epdf_points, sizeof(double));
	cdf = calloc(epdf_points, sizeof(double));
	pdf_predicted = calloc(epdf_points, sizeof(double));

	printf("Using %zu points for PDF/EPDF plotting\n", epdf_points);

	outfile = fopen("mixtures_rnd.dat", "w");

	if (!outfile) {
		perror("fopen");
		retval = EXIT_FAILURE;
		goto out;
	}

	printf("Writing sorted records... ");

	for (i = 0; i < num_points; i++)
		fprintf(outfile, "%lf\n", rnd[i]);

	fclose(outfile);
	printf("done\n");

	printf("Activating GPU... ");
	// stable_activate_gpu(dist);
	printf("done\n");

	printf("Plotting initial EPDF... ");
	fflush(stdout);
	outfile = fopen("mixtures_dat.dat", "w");

	for (i = 0; i < epdf_points; i++)
		x[i] = mn + i * (mx - mn) / epdf_points;

	if (has_real_pdf)
		stable_pdf(dist, x, epdf_points, pdf, NULL);
	else
		memset(pdf, 0, sizeof(double) * epdf_points);

	printf("done\n");

	printf("Starting mixture estimation.\n");

	// Adjust settings for the estimation. Read the structure stable_mcmc_settings documentation in stable_api.h
	// for details on what these mean.
	dist->max_mixture_components = 5;
	stable_fit_mixture_default_settings(&settings);
	settings.max_iterations = 25000;
	settings.fix_components_during_last_n_iterations = 5000;
	settings.fix_components_during_first_n_iterations = 0;
	settings.skip_initial_estimation = 0;
	settings.location_lock_iterations = 0;
	settings.fix_components = 0;
	settings.force_gaussian = 0;
	settings.force_cauchy = 1;
	settings.force_full_epdf_range = 1;
	settings.generator_variance_ms = 0.5;
	settings.default_bd_prob = 0.05;
	settings.weight_prior = 1000;

	struct gaussian_params mu_params = { .mean = 5, .variance = 0.5};
	//settings.prior_functions[STABLE_PARAM_MU] = (prior_probability) _mixture_gaussian_prior;
	//settings.prior_parameters[STABLE_PARAM_MU] = &mu_params;

	summary = fopen("mixtures_summary.txt", "w");

	stable_fit_mixture_settings(dist, rnd, num_points, &settings);
	stable_fit_mixture_print_results(&settings, stdout);
	stable_fit_mixture_print_results(&settings, summary);

	if (settings.num_samples > 0) {
		stable_pdf(dist, x, epdf_points, pdf_predicted, NULL);
		stable_cdf(dist, x, epdf_points, cdf, NULL);
	}

	printf("Writing EPDF files\n");

	for (i = 0; i < epdf_points; i++)
		fprintf(outfile, "%lf %lf %lf %lf\n", x[i], pdf[i], pdf_predicted[i], cdf[i]);

	fclose(outfile);
	fclose(summary);

	printf("Done\n");
out:
	stable_free(dist);

	clock_gettime(CLOCK_MONOTONIC, &t_end);

	double runtime_sec = (t_end.tv_sec - t_start.tv_sec);
	int runtime_onlysec = fmod(runtime_sec, 60);
	int runtime_min = fmod(runtime_sec / 60, 60);
	int runtime_hours = runtime_sec / 3600;

	printf("Total time: %dh%dm%ds (%.0lf seconds total)\n", runtime_hours, runtime_min, runtime_onlysec, runtime_sec);

	return retval;
}
