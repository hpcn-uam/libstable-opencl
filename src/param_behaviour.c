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

#include "stable_api.h"
#include "benchmarking.h"
#include "sysutils.h"
#include "methods.h"
#include <time.h>
#include <stdlib.h>

#define ALFA_START 0.3
#define ALFA_END 2
#define ALPHA_INCR 0.02
#define BETA_START -1
#define BETA_END 1
#define BETA_INCR 0.02
#define SIGMA_INCR 0.25
#define SIGMA_START SIGMA_INCR
#define SIGMA_END 2


int main(int argc, char *argv[])
{
	double alfa, beta, sigma, mu_0;
	double *x;
	size_t nx = 10000;
	double pdf[nx];
	double x_start = -5;
	double x_end = 5;
	double x_step;
	size_t max_pos, pos;
	double max_value, max_x;
	double left_deriv_1, left_deriv_2, left_deriv_3, right_deriv_1, right_deriv_2, right_deriv_3;
	double left_x_1, left_x_2, left_x_3, right_x_1, right_x_2, right_x_3;
	double peakdst;
	FILE* fout;

	StableDist *dist = NULL;

	alfa = 1.5;
	beta = 0.75;
	sigma = 5.0;
	mu_0 = 15.0;

	install_stop_handlers();

	if ((dist = stable_create(alfa, beta, sigma, mu_0, 0)) == NULL) {
		printf("Error when creating the distribution");
		exit(1);
	}

	stable_set_THREADS(8);
	stable_set_absTOL(1e-2);
	stable_set_relTOL(1e-2);

	vector_npoints(&x, x_start, x_end, nx, &x_step);

	stable_activate_gpu(dist);

	fout = fopen("param_behaviour.dat", "w");

	for (alfa = ALFA_START; alfa <= ALFA_END + 2 * DBL_EPSILON; alfa += ALPHA_INCR) {
		for (beta = BETA_START; beta <= BETA_END + 2 * DBL_EPSILON; beta += BETA_INCR) {
			for (sigma = SIGMA_START; sigma <= SIGMA_END + 2 * DBL_EPSILON; sigma += SIGMA_INCR) {
				mu_0 = 0;

				printf("\rα = %lf, β = %lf, σ = %lf", alfa, beta, sigma);
				fflush(stdout);

				stable_setparams(dist, alfa, beta, sigma, mu_0, 0);
				stable_pdf_gpu(dist, x, nx, pdf, NULL);

				max_pos = find_max(pdf, nx);
				max_value = pdf[max_pos];
				max_x = x[max_pos];

				left_deriv_1 = get_derivative_at_pctg_of_max(pdf, max_pos, max_value, 0.95, x_step, 1, &pos);
				left_x_1 = x[pos];
				right_deriv_1 = get_derivative_at_pctg_of_max(pdf + max_pos, nx - max_pos, max_value, 0.95, x_step, 0, &pos);
				right_x_1 = x[pos + max_pos];

				left_deriv_2 = get_derivative_at_pctg_of_max(pdf, max_pos, max_value, 0.75, x_step, 1, &pos);
				left_x_2 = x[pos];
				right_deriv_2 = get_derivative_at_pctg_of_max(pdf + max_pos, nx - max_pos, max_value, 0.75, x_step, 0, &pos);
				right_x_2 = x[pos + max_pos];

				left_deriv_3 = get_derivative_at_pctg_of_max(pdf, max_pos, max_value, 0.85, x_step, 1, &pos);
				left_x_3 = x[pos];
				right_deriv_3 = get_derivative_at_pctg_of_max(pdf + max_pos, nx - max_pos, max_value, 0.85, x_step, 0, &pos);
				right_x_3 = x[pos + max_pos];

				peakdst = (max_x - left_x_1) - (right_x_1 - max_x);

				fprintf(fout, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
						alfa, beta, sigma, left_deriv_1, right_deriv_1,
						left_deriv_2, right_deriv_2,
						right_x_1 - left_x_1, right_x_2 - left_x_2, peakdst, max_value, max_x,
						left_deriv_3, right_deriv_3, right_x_3 - left_x_3);
			}
		}
	}


	printf("\nEnd");
	free(x);
	fclose(fout);
	stable_free(dist);

	return 0;
}
