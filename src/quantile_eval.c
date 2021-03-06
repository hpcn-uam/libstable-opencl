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

int main(int argc, const char** argv)
{
	double alfas[] = { 0.25, 0.5, 0.75, 1.25, 1.5 };
	double betas[] = { 0, 0.5, 1 };
	double intervals[] = { -10, 10 };
	int points_per_interval = 20;
	double cdf_vals[points_per_interval];
	double guesses[points_per_interval];
	short use_all_gpu = 1;

	stable_clinteg_printinfo();

	StableDist *dist = stable_create(0.5, 0, 1, 0, 0);

	if (!dist) {
		fprintf(stderr, "StableDist creation failure. Aborting.\n");
		return 1;
	}

	if (stable_activate_gpu(dist)) {
		fprintf(stderr, "Couldn't initialize GPU.\n");
		return 1;
	}

	if (argc > 1)
		use_all_gpu = 0;

	stable_set_absTOL(1e-20);
	stable_set_relTOL(1.2e-10);

	size_t ai, bi, i, j;
	double points[points_per_interval];
	size_t interval_count = (sizeof(intervals) / sizeof(double)) - 1;
	size_t alfa_count = sizeof(alfas) / sizeof(double);
	size_t beta_count = sizeof(betas) / sizeof(double);
	double total_relerr = 0, total_abserr = 0;
	size_t valid_points;

	double abs_diff_sum, rel_diff_sum, gpu_err_sum, cpu_err_sum;

	for (i = 0; i < interval_count; i++) {
		double begin = intervals[i];
		double end = intervals[i + 1];
		double step = (end - begin) / points_per_interval;

		printf("\n=== Interval (%.0lf, %.0lf)\n", begin, end);
		printf("alfa  beta   abserr  relerr\n");

		for (j = 0; j < points_per_interval; j++)
			points[j] = j * step + begin;

		for (ai = 0; ai < alfa_count; ai++) {
			for (bi = 0; bi < beta_count; bi++) {
				stable_setparams(dist, alfas[ai], betas[bi], 1, 0, 0);
				stable_cdf_gpu(dist, points, points_per_interval, cdf_vals, NULL);

				abs_diff_sum = 0;
				rel_diff_sum = 0;
				gpu_err_sum = 0;
				cpu_err_sum = 0;
				valid_points = 0;

				if (use_all_gpu)
					stable_inv_gpu(dist, cdf_vals, points_per_interval, guesses, NULL);

				for (j = 0; j < points_per_interval; j++) {
					if (cdf_vals[j] >= 0.1 && cdf_vals[j] <= 0.9) {
						double guess;

						if (use_all_gpu)
							guess = guesses[j];
						else
							guess = stable_inv_point_gpu(dist, cdf_vals[j], NULL);

						if (isnan(guess))
							continue;

						double diff = fabs(guess - points[j]);

						abs_diff_sum += diff;

						if (points[j] != 0)
							rel_diff_sum += diff / points[j];

						valid_points++;
					}
				}

				total_relerr += rel_diff_sum;
				total_abserr += abs_diff_sum;

				if (valid_points != 0) {
					abs_diff_sum /= valid_points;
					rel_diff_sum /= valid_points;
				}

				printf("%.3lf %.3lf  %8.3g %8.3g\n",
					   alfas[ai], betas[bi],
					   abs_diff_sum, rel_diff_sum);
			}
		}
	}


	stable_free(dist);
	return 0;
}
