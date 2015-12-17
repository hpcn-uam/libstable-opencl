/* stable/stable_pdf.c
 *
 * Functions wrappers of GSL routines for random sample generation of
 * alpha-stable random variable.
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
#include "opencl_integ.h"
#include <gsl/gsl_randist.h>

void
stable_rnd_seed(StableDist * dist, unsigned long int s)
{
	gsl_rng_set(dist->gslrand, s);
}


inline double
stable_rnd_point(StableDist *dist)
{
	return dist->mu_1 +
	       gsl_ran_levy_skew(dist->gslrand, dist->sigma, dist->alfa, dist->beta);
}

void
stable_rnd(StableDist *dist, double *rnd, const unsigned int n)
{
	//double *rnd;
	int i;

	//rnd = (double*)malloc(n*sizeof(double));
	if (rnd == NULL) exit(2);

	for (i = 0; i < n; i++)
	{
		rnd[i] = stable_rnd_point(dist);
	}
	return;
}

short stable_rnd_gpu(StableDist *dist, double *rnd, const unsigned int n)
{
	stable_clinteg_set_mode(&dist->cli, mode_rng);

    return stable_clinteg_points(&dist->cli, NULL, rnd, NULL, NULL, n, dist);
}


