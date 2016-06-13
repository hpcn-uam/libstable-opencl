/**
 * @file kde.h
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



#include <stdio.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_math.h>

double kerneldensity(double *samples, double obs, size_t n);