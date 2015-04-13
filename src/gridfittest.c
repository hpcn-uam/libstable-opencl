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

int main (int argc, char *argv[])
{
	double alfa, beta, sigma, mu_0;
	double *data;
	int i = 1, iexp, N, Nexp;
	int seed;
	double acc_pdev;
	double total_duration, start, end;
	double *pdf;
	double ms_duration;

	StableDist *dist = NULL;

	alfa = 1.5;
	beta = 0.75;
	sigma = 5.0;
	mu_0 = 15.0;
	N = 400;
	Nexp = 1;
	seed = -1;

	printf("Parameters for the random data generated:\n");
	printf("α\t%lf\n", alfa);
	printf("β\t%lf\n", beta);
	printf("σ\t%lf\n", sigma);
	printf("μ\t%lf\n", mu_0);
	printf("Size\t%d\n", N);
	printf("\nWill perform %d experiments.\n\n", Nexp);

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
	data = (double *)malloc(N * sizeof(double));
	pdf = (double *) calloc(N, sizeof(double));

	stable_rnd(dist, data, N);

	total_duration = 0;

	stable_fit_init(dist, data, N, NULL, NULL);

	stable_activate_gpu(dist);

	start = get_ms_time();
	stable_fit_grid(dist, data, N);
	end = get_ms_time();

	ms_duration = end - start;


	printf("time = %lf\nα = %lf\nβ = %.2lf\nμ = %.2lf\nσ = %.2lf\n",
	       ms_duration, dist->alfa, dist->beta, dist->sigma, dist->mu_0);

	free(data);
	stable_free(dist);

	fclose(stable_get_FLOG());

	return 0;
}
