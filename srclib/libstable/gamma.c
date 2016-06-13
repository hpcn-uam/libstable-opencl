#include "gamma.h"

#include <math.h>
#include <gsl/gsl_sf_gamma.h>

double invgamma_pdf(double alpha, double beta, double x)
{
	return exp(alpha * log(beta) - gsl_sf_gamma(alpha) + (alpha + 1) * log(1 / x) - beta / x);
}
