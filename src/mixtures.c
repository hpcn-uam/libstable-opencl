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

#define MAX_POINTS 50000

int main(int argc, char **argv)
{
	size_t num_points = 5000;
	size_t epdf_points = 1000;
	size_t i;

	assert(MAX_POINTS >= num_points);

	/*
	double alphas[] = { 1.2, 0.8, 2 };
	double betas[] = { -0.5, 0.5, 0 };
	double mus[] = { -2, 0, 2 };
	double sigmas[] = { 0.5, 0.8, 0.2 };
	double weights[] = { 0.2, 0.5, 0.3 };
	*/

	/*
	double alphas[] = { 0.35, 0.6 };
	double betas[] = { 0.8, 0 };
	double mus[] = { 1.5, 1.65 };
	double sigmas[] = { 0.05, 0.05 };
	double weights[] = { 0.7, 0.3 };
	*/


	double alphas[] = { 1.2, 0.8, 2, 0.6 };
	double betas[] = { -0.5, 0.5, 0, 0 };
	double mus[] = { -2, 0, 2, 2.5 };
	double sigmas[] = { 0.5, 0.8, 0.15, 0.15 };
	double weights[] = { 0.2, 0.3, 0.3, 0.2 };

	size_t num_components = sizeof weights / sizeof(double);

	double rnd[MAX_POINTS];
	double pdf[epdf_points];
	double pdf_predicted[epdf_points];
	double x[epdf_points];
	double epdf[epdf_points], epdf_finer[epdf_points];
	double mn = -5, mx = 5;
	short has_real_pdf = 0;

	FILE* infile = NULL;
	FILE* outfile;
	int retval = EXIT_SUCCESS;

	StableDist* dist;

	if (argc == 2) {
		infile = fopen(argv[1], "r");

		if (!infile) {
			perror("fopen");
			return EXIT_FAILURE;
		}
	}

	gsl_set_error_handler_off();

	dist = stable_create(alphas[0], betas[0], sigmas[0], mus[0], 0);

	if (!dist) {
		fprintf(stderr, "StableDist creation failure. Aborting.\n");
		return 1;
	}

	if (infile == NULL) {
		printf("Generating random numbers...\n");
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

	printf("Sorting records... ");
	gsl_sort(rnd, 1, num_points);
	printf("done\n");

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

	printf("Plotting initial EPDF... ");
	fflush(stdout);
	outfile = fopen("mixtures_dat.dat", "w");

	for (i = 0; i < epdf_points; i++) {
		x[i] = mn + i * (mx - mn) / num_points;
		epdf[i] = kerneldensity(rnd, x[i], num_points, MIXTURE_KERNEL_ADJUST);
		epdf_finer[i] = kerneldensity(rnd, x[i], num_points, MIXTURE_KERNEL_ADJUST_FINER);
	}

	printf("done\n");

	stable_activate_gpu(dist);

	if (has_real_pdf)
		stable_pdf(dist, x, epdf_points, pdf, NULL);
	else
		memset(pdf, 0, sizeof(double) * epdf_points);

	printf("Starting mixture estimation.\n");
	stable_fit_mixture(dist, rnd, num_points);

	stable_pdf(dist, x, epdf_points, pdf_predicted, NULL);

	for (i = 0; i < epdf_points; i++)
		fprintf(outfile, "%lf %lf %lf %lf %lf\n", x[i], pdf[i], pdf_predicted[i], epdf[i], epdf_finer[i]);

	fclose(outfile);

	printf("Done\n");
out:
	stable_free(dist);

	return retval;
}
