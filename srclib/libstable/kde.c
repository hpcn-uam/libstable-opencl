/**
 * @file kde.c
 * @author Carl Boettiger, <cboettig@gmail.com>
 * @section DESCRIPTION
 * Estimates the kernel density p(x) at a given value x from
 * an array of sample points.  Uses the default algorithm from
 * the R langauge's 'density' function.  Requires the GSL statistics
 * library.  
 *   
 * @section LICENCE
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 3 of
 * the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details at
 * http://www.gnu.org/copyleft/gpl.html
 *
 */


#include "kde.h"

/** Estimate bandwidth using Silverman's "rule of thumb" 
 * (Silverman 1986, pg 48 eq 3.31).  This is the default
 * bandwith estimator for the R 'density' function.  */
double nrd0(double x[], const int N)
{
	gsl_sort(x, 1, N);
	double hi = gsl_stats_sd(x, 1, N);
	double iqr =
		gsl_stats_quantile_from_sorted_data (x,1, N,0.75) - 
        gsl_stats_quantile_from_sorted_data (x,1, N,0.25);
	double lo = GSL_MIN(hi, iqr/1.34);
	double bw = 0.9 * lo * pow(N,-0.2);
	return(bw);
}

/* kernels for kernel density estimates */
double gauss_kernel(double x)
{ 
	return exp(-(gsl_pow_2(x)/2))/(M_SQRT2*sqrt(M_PI)); 
}

double kerneldensity(double *samples, double obs, size_t n)
{
	size_t i;
	double h = GSL_MAX(nrd0(samples, n), 1e-6);
	double prob = 0;
	for(i=0; i < n; i++)
	{
		prob += gauss_kernel( (samples[i] - obs)/h)/(n*h);
	}
	return prob;
}

