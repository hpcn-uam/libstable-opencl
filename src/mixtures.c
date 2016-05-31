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
#include "stable_api.h"
#include "benchmarking.h"
#include "opencl_integ.h"

int main(int argc, char **argv)
{
	size_t num_points = 10000;
	size_t num_components = 3;
	size_t i;
	double alphas[] = { 1.2, 0.8, 2 };
	double betas[] = { -0.5, 0.5, 0 };
	double mus[] = { -2, 0, 2 };
	double sigmas[] = { 0.5, 0.8, 0.2 };
	double weights[] = { 0.2, 0.5, 0.3 };
	double rnd[num_points];
	FILE* outfile;
	int retval = EXIT_SUCCESS;

	StableDist* dist;

	if (argc == 2)
		num_points = strtod(argv[1], NULL);

	dist = stable_create(alphas[0], betas[0], sigmas[0], mus[0], 0);

	if (!dist) {
		fprintf(stderr, "StableDist creation failure. Aborting.\n");
		return 1;
	}

	stable_set_mixture_components(dist, num_components);

	for (i = 0; i < num_components; i++) {
		dist->mixture_weights[i] = weights[i];
		stable_setparams(dist->mixture_components[i], alphas[i], betas[i], sigmas[i], mus[i], 0);
	}

	stable_rnd(dist, rnd, num_points);

	outfile = fopen("mixtures_rnd.dat", "w");

	if (!outfile) {
		perror("fopen");
		retval = EXIT_FAILURE;
		goto out;
	}

	for (i = 0; i < num_points; i++)
		fprintf(outfile, "%lf\n", rnd[i]);


	printf("Done\n");
out:
	stable_free(dist);

	return retval;
}
