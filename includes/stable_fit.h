#ifndef _STABLE_FIT_H_
#define _STABLE_FIT_H_

#include "stable_api.h"


gsl_complex stable_samplecharfunc_point(const double x[],
										const unsigned int N, double t);

void stable_samplecharfunc(const double x[], const unsigned int Nx,
						   const double t[], const unsigned int Nt, gsl_complex *z);

#endif
