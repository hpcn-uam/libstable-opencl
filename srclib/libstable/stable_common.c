/* stable/stable_common.c
 *
 * Library global parameters and default values
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

#include <stdio.h>
#include "stable_api.h"

#define TINY 1e-50
#define EPS 2.2204460492503131E-16

FILE * FLOG = NULL;            // Log file (optional)
FILE * FINTEG = NULL;          // Integrand evaluations output file (debug purposes)

unsigned short THREADS = 0;    // threads of execution (0 => total available)
unsigned short IT_MAX = 1000;  // Maximum # of iterations in quadrature methods
unsigned short SUBS = 3;       // # of integration subintervals
unsigned short METHOD = STABLE_QNG;  // Integration method on main subinterval
unsigned short METHOD2 = STABLE_QAG2; // Integration method on second subinterval
unsigned short METHOD3 = STABLE_QUADSTEP; // Integration method on third subinterval
unsigned short METHOD_ = STABLE_QNG; // Default integration method

unsigned short INV_MAXITER = 15; // Maximum # of iterations inversion method

double relTOL = 1e-6;        // Relative error tolerance
double absTOL = 1e-6;        // Absolut error tolerance
//double FACTOR = 50*EPS;
double ALFA_TH = 1.0e-3;     // Alfa threshold
double BETA_TH = 1.0e-3;     // Beta threshold
double EXP_MAX = -200;       // Exponent maximum value
double XXI_TH = 0.00001;     // Zeta threshold
double THETA_TH = 10 * EPS;  // Theta threshold

double AUX1 = -13.81551056; //log(1e-6) // Auxiliary values
double AUX2 =  2.91099795; //log(18)

#ifdef DEBUG
unsigned int integ_eval = 0; // # of integrand evaluations
#endif

void stable_log(const char *text)
{
	if (FLOG)
		fprintf(FLOG, "%s", text);
	else printf("%s", text);
}

void error_handler(const char * reason, const char * file,
				   int line, int gsl_errno)
{
	char message[1024];

	snprintf(message, 1024, "%s: %d: ERROR #%d: %s\n", file, line, gsl_errno, reason);
	stable_log(message);
}

void stable_swap_components(StableDist* dist, size_t comp1, size_t comp2)
{
	StableDist* swap = dist->mixture_components[comp1];
	double wswap = dist->mixture_weights[comp1];

	dist->mixture_components[comp1] = dist->mixture_components[comp2];
	dist->mixture_components[comp2] = swap;
	dist->mixture_weights[comp1] = dist->mixture_weights[comp2];
	dist->mixture_weights[comp2] = wswap;
}
