/* stable/methods.h
 *
 * Numerical methods employed by Libstable. Some methods extracted from:
 *  Press, W.H. et al. Numerical Recipes in C: the art of scientific
 *    computing. Cambridge University Press, 1994. Cambridge, UK.
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
#ifndef _METHODS_H_
#define _METHODS_H_

#define MAX(a,b,c,d) ( a > b ? c : d)

#include "mcculloch.h"
#include <stddef.h>

double gammaln(const double x);

double zbrent(double (*func)(double x, void *args), void * args, double x1, double x2, double value, const double tol, int *warn);

double dfridr(double (*func)(double, void * args), void * args, double x,  double h, double *err);

double quadstep(double (*func)(double x, void *args), void * args, double a, double b, double fa, double fc, double fb, const double epsabs, const double epsrel, double *abserr, int *warn, size_t *fcnt);

double qromb(double (*func)(double, void *), void *args, double a, double b, double epsabs, double epsrel, int K, int JMAX, int method, int *warn, size_t *fcnt, double *err);

double trapzd(double (*func)(double, void *), void * args, double a, double b, int n, double s);

void polint(const double xa[], const double ya[], const int n, double x, double *y, double *dy);

void ratint(const double xa[], const double ya[], const int n, double x, double *y, double *dy);

void medfilt(const double xa[], const double ya[], int n, double N);

void vector_step(double **x, double min, double max, double step, int *n);

void vector_npoints(double **x, double min, double max, int n, double * step);

/**
 * Does a binary search, returning the nearest element to the searched one.
 * @param  data     Array with the numbers.
 * @param  length   Length of the data array.
 * @param  value    Value to search.
 * @param  round_up If 1, return the nearest element greater than value. If 0, the nearest
 *                  and less than value.
 * @return          Position in the array of the nearest element.
 */
size_t binary_search_nearest(double* data, size_t length, double value, short round_up);

/** Same as the above method, but with data sorted in descending order. */
size_t binary_search_nearest_desc(double* data, size_t length, double value, short round_up);

/**
 * Return the inverse Gamma PDF at the given point.
 * @param  alpha Parameter alpha of the IG distr.
 * @param  beta  Parameter beta of the IG distr.
 * @param  x     Point at which the PDF should be evaluated.
 * @return       PDF value.
 */
double invgamma_pdf(double alpha, double beta, double x);

/**
 * Return the position of the maximum value in the array
 * @param  data   Data array
 * @param  length Length of the array
 * @return        Position of the maximum value
 */
size_t find_max(double* data, size_t length);

/**
 * Reverse a vector
 * @param src      Source vector
 * @param reversed Where to store the reversed vector
 * @param length   Length of the vector.
 */
void reverse(double* src, double* reversed, size_t length);

/**
 * Given a monomodal function (e.g, a PDF), find the value of the derivative when the value is
 * at a given percentage of the maximum
 * @param  pdf       Values of the function. Remember to pass only the right or left part to avoid
 *                   	duplicate values.
 * @param  npoints   Number of sampled points of the function
 * @param  max       Maximum of the function
 * @param  pctg      Percentage for which to find the derivative
 * @param  x_step    Step between points of the pdf
 * @param  left_part Whether we are searching to the left of the maximum (ie, pdf is ordered in ascending
 *                   	order) or to the right (pdf in descending order).
 * @param  pos       If not NULL, save in this pointer the location of the derivative.
 *
 * @return           Value of the derivative.
 */
double get_derivative_at_pctg_of_max(double* pdf, size_t npoints, double max, double pctg, double x_step, short left_part, size_t* pos);

/**
 * Calculate an empirical PDF based on the given samples.
 * @param samples   Array with the samples.
 * @param nsamples  Number of samples in the array.
 * @param start_x 	X value where the EPDF starts
 * @param end_x     X value where the EPDF ends
 * @param npoints   Number of points to sample for in the EPDF.
 * @param bw_adjust Bandwidth adjustment factor.
 * @param epdf_x    If not NULL, store here the X values of the EPDF. Must have length npoints
 * @param epdf      Store here the values of the EPDF.
 */
void calculate_epdf(const double* samples, size_t nsamples, double start_x, double end_x, size_t npoints, double bw_adjust, double* epdf_x, double* epdf);
#endif
