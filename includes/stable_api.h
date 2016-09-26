/* stable/stable_api.h
 *
 * Main header file of Libstable. Contains all declarations of the
 * usable functions in the library.
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
#ifndef _STABLE_API_H_
#define _STABLE_API_H_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <gsl/gsl_integration.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_vector.h>

#include "opencl_integ.h"

#define TINY 1e-50
#define EPS 2.2204460492503131E-16

#define max(a,b) (((a) > (b)) ? (a) : (b))
#define min(a,b) (((a) < (b)) ? (a) : (b))

/******************************************************************************/
/*          Library parameters                                                */
/******************************************************************************/

extern FILE * FLOG;            // Log file (optional)
extern FILE * FINTEG;          // Integrand evaluations output file (debug purposes)

extern unsigned short THREADS;    // threads of execution (0 => total available)
extern unsigned short IT_MAX;     // Maximum # of iterations in quadrature methods
extern unsigned short SUBS;       // # of integration subintervals
extern unsigned short METHOD;     // Integration method on main subinterval
extern unsigned short METHOD2;    // Integration method on second subinterval
extern unsigned short METHOD3;    // Integration method on third subinterval
extern unsigned short METHOD_;    // Default integration method

extern unsigned short INV_MAXITER; // Maximum # of iterations inversion method

extern double relTOL;     // Relative error tolerance
extern double absTOL;     // Absolut error tolerance
//extern double FACTOR;   //
extern double ALFA_TH;    // Alfa threshold
extern double BETA_TH;    // Beta threshold
extern double EXP_MAX;    // Exponent maximum value
extern double XXI_TH;     // Zeta threshold
extern double THETA_TH;   // Theta threshold

extern double AUX1; // Auxiliary values
extern double AUX2;

extern double MIXTURE_KERNEL_ADJUST; // Adjust value for Silverman's rule of thumb (affects maximum detection in the mixture preparation)
extern double MIXTURE_KERNEL_ADJUST_FINER; // Adjust value for Silverman's rule of thumb (affects maximum detection in the mixture preparation)

#ifdef DEBUG
extern unsigned int integ_eval; // # of integrand evaluations
#endif

/******************************************************************************/
/******************************************************************************/
// Particular cases
enum {
	NOVALID = -1,
	STABLE,
	ALFA_1,
	GAUSS ,
	CAUCHY,
	LEVY,
	STABLE_B1,
	ALFA_1_B1
};

// Function to evaluate
enum {
	CDF,
	PDF
};

// Quadrature methods
enum {
	STABLE_QAG2 = 0,
	STABLE_QUADSTEP,
	STABLE_QROMBPOL,
	STABLE_QROMBRAT,
	STABLE_QNG,
	STABLE_QAG1,
	STABLE_QAG5,
	STABLE_VECT,
	STABLE_OCL
};

// Index of the parameters
enum {
	STABLE_PARAM_ALPHA = 0,
	STABLE_PARAM_BETA = 1,
	STABLE_PARAM_MU = 2,
	STABLE_PARAM_SIGMA = 3
};

#define MAX_STABLE_PARAMS 4



/************************************************************************
 ************************************************************************
 * Scalar methods                                                       *
 ************************************************************************
 ************************************************************************/
/* Scalar methods are those that, for each thread of execution, obtain a
   single evaluation of the desired function, at a single point.
*/

/******************************************************************************/
/*    Stable distribution structure.                                          */
/******************************************************************************/
struct StableDistStruct {
	/* Parameters:
	0-parametrization describen in Nolan, 1997 is employed by default
	    alfa : stability index
	    beta : skewness parameter
	    scale: scale parameter
	    mu_0 : 0-parametrization location parameter
	    mu_1 : correspondig 1-parametrization location parameter    */
	double alfa;
	double beta;
	double sigma;
	double mu_0;
	double mu_1;

	short is_mixture;
	struct StableDistStruct** mixture_components;
	double* mixture_weights;
	size_t num_mixture_components;
	size_t max_mixture_components;
	double mixture_montecarlo_variance;
	size_t allocated_mixture_components;

	double prior_mu_avg;
	double prior_mu_variance;
	double prior_sigma_alpha0;
	double prior_sigma_beta0;
	double prior_weights;

	double* birth_probs;
	double* death_probs;

	/* Particular cases indicator (Gauss, Cauchy, Levy distribution, alfa==1, etc.) */
	int ZONE;

	/* Pointers to pdf and cdf evaluation functions */
	double(*stable_pdf_point)(struct StableDistStruct *, const double, double *);
	double(*stable_cdf_point)(struct StableDistStruct *, const double, double *);

	/* Precalculated values. */
	double alfainvalfa1;  /* alfa/(alfa-1)*/
	double xi;            /* -beta*tan(alfa*pi/2)*/
	double theta0;        /* 1/alfa*atan(beta*(tan(alfa*pi/2))=atan(-xi)/alfa;*/
	double c1, c2_part, c3;  /* additive and multiplicative constants*/
	double k1;     /* cos(alfa*theta0)^(1/(alfa-1)) = (1+xi^2)^(-0.5/(alfa-1));*/
	double S;     /* (1+xi^2)^(1/(2*alfa));*/
	double Vbeta1; /*pow(1/dist->alfa,dist->alfainvalfa1) *
                     (dist->alfa-1)*pow(-cos(dist->alfa*PI_2),1/(dist->alfa-1))*/

	/* These ones change from point to point of evaluation */
	double theta0_; /* theta0_ = +-theta0 */
	double beta_;
	double xxipow;  /* (x-xi)^(alfa/(alfa-1))*/

	/* gsl integration workspace */
	gsl_integration_workspace * gslworkspace;

	/* gsl random numbers generator */
	gsl_rng * gslrand;

	struct stable_clinteg cli;
	short gpu_enabled;
	short parallel_gridfit;
	size_t gpu_platform;
	size_t gpu_queues;
};

typedef struct StableDistStruct StableDist;

/** Function pointer for array evaluator (e.g., stable_pdf, stable_cdf) */
typedef void(*array_evaluator)(StableDist *, const double*, const int, double*, double*);

/******************************************************************************/
/*        Auxiliary functions                                                 */
/******************************************************************************/
unsigned int stable_get_THREADS();
void   stable_set_THREADS(unsigned int threads);

unsigned int stable_get_IT_MAX();
void   stable_set_IT_MAX(unsigned int itmax);

unsigned int stable_get_INV_MAXITER();
void   stable_set_INV_MAXITER(unsigned int invmaxiter);

int stable_get_METHOD();
void stable_set_METHOD(int method);

int stable_get_METHOD2();
void stable_set_METHOD2(int method);

int stable_get_METHOD3();
void stable_set_METHOD3(int method);

double stable_get_relTOL();
void   stable_set_relTOL(double reltol);

double stable_get_absTOL();
void   stable_set_absTOL(double abstol);

double stable_get_ALFA_TH();
void   stable_set_ALFA_TH(double alfath);

double stable_get_BETA_TH();
void   stable_set_BETA_TH(double betath);

double stable_get_XXI_TH();
void   stable_set_XXI_TH(double xxith);

double stable_get_THETA_TH();
void   stable_set_THETA_TH(double thetath);

FILE * stable_get_FINTEG();
FILE * stable_set_FINTEG(char * filename);

FILE * stable_get_FLOG();
FILE * stable_set_FLOG(char * filename);


StableDist *stable_create(double alfa, double beta, double sigma, double mu,
						  int parametrization);

short stable_activate_gpu(StableDist* dist);
void stable_deactivate_gpu(StableDist* dist);

void stable_print_params_array(double params[4], const char* prefix, ...);
void stable_print_params(StableDist* dist, const char* prefix, ...);

/**
 * Enable the mixture and set the given number of components or, if the number is
 * zero, disable the mixture.
 *
 * @param  dist           Stable distribution.
 * @param  num_components Number of components in the mixture.
 */
short stable_set_mixture_components(StableDist* dist, size_t num_components);
short stable_disable_mixture(StableDist* dist);

StableDist *stable_copy(StableDist *src_dist);

void stable_free(StableDist *dist);

int stable_setparams(StableDist *dist,
					 double alfa, double beta, double sigma, double mu,
					 int parametrization);

int stable_checkparams(double alfa, double beta, double sigma, double mu,
					   int parametrization);

void error_handler(const char * reason, const char * file,
				   int line, int gsl_errno);

/**
 * Gets the parameters of the distribution in an array.
 *
 * Useful for iterating through the parameters programatically.
 */
void stable_getparams_array(StableDist* dist, double params[4]);

/**
 * Sets the parameter for the distirbution using the ones of the array.
 *
 * Useful for iterating through the parameters programatically.
 *
 * @return  0 if the parameters are valid, 1 if not
 */
short stable_setparams_array(StableDist* dist, double params[4]);

/******************************************************************************/
/*   PDF in particular cases                                                  */
/******************************************************************************/

double stable_pdf_point_GAUSS(StableDist *dist, const double x, double *err);

double stable_pdf_point_CAUCHY(StableDist *dist, const double x, double *err);

double stable_pdf_point_LEVY(StableDist *dist, const double x, double *err);

/******************************************************************************/
/*   PDF otherwise                                                            */
/******************************************************************************/

double stable_pdf_point_STABLE(StableDist *dist, const double x, double *err);

double stable_pdf_point_ALFA_1(StableDist *dist, const double x, double *err);

double stable_pdf_point(StableDist *dist, const double x, double *err);

void stable_pdf(StableDist *dist, const double x[], const int Nx,
				double *pdf, double *err);

void stable_pdf_gpu(StableDist *dist, const double x[], const int Nx,
					double *pdf, double *err);


void stable_pcdf_gpu(StableDist *dist, const double x[], const int Nx,
					 double *pcdf, double *cdf);

/******************************************************************************/
/*   PDF integrand functions                                                  */
/******************************************************************************/

double stable_pdf_g(double theta, void *dist);
double stable_pdf_g_aux1(double theta, void *args);
double stable_pdf_g_aux2(double theta, void *args);

/******************************************************************************/
/*   CDF in particular cases                                                  */
/******************************************************************************/

double stable_cdf_point_GAUSS(StableDist *dist, const double x, double *err);

double stable_cdf_point_CAUCHY(StableDist *dist, const double x, double *err);

double stable_cdf_point_LEVY(StableDist *dist, const double x, double *err);

/******************************************************************************/
/*   CDF otherwise                                                            */
/******************************************************************************/

double stable_cdf_point_STABLE(StableDist *dist, const double x, double *err);

double stable_cdf_point_ALFA_1(StableDist *dist, const double x, double *err);

double stable_cdf_point(StableDist *dist, const double x, double *err);

void stable_cdf(StableDist *dist, const double x[], const int Nx,
				double *cdf, double *err);

void stable_cdf_gpu(StableDist *dist, const double x[], const int Nx,
					double *cdf, double *err);

/******************************************************************************/
/*   CDF integrad functions                                                   */
/******************************************************************************/

double stable_cdf_g(double theta, void *dist);

/******************************************************************************/
/*   CDF^{-1} (quantiles)                                                     */
/******************************************************************************/

double stable_inv_point(StableDist * dist, const double q, double * err);
void   stable_inv(StableDist *dist, const double q[], const int Nq,
				  double * inv, double * err);
double stable_inv_point_gpu(StableDist* dist, const double q, double *err);
short stable_inv_gpu(StableDist *dist, const double q[], const int Nq,
					 double *inv, double *err);

/************************************************************************
 ************************************************************************
 * Vectorial methods                                                    *
 ************************************************************************
 ************************************************************************/
/*
  Alternative non-parallelized methods of evaluation have been implemented.
  These methods exploit the fact that some calculations are shared between
  different points of evaluation when evaluating the PDF or CDF, so these
  calculations can be realized just once.

  The performance achieved is high, sometimes comparable with parallel
  methods when little precision is required. However, achievable precision
  with these methods is low and non desired behavior of the PDF and CDF
  evaluation is observed.
*/

/* Stable distribution structure for vectorial methods*/

typedef struct {
	/* Parameters:
	    0-parametrization describen in Nolan, 1997 is employed by default
	        alfa : stability index
	        beta : skewness parameter
	        scale: scale parameter
	        mu_0 : 0-parametrization location parameter
	        mu_1 : correspondig 1-parametrization location parameter    */
	double alfa;
	double beta;
	double sigma;
	double mu_0;
	double mu_1;

	/* Particular cases indicator (Gauss, Cauchy, Levy distribution, alfa==1, etc.) */
	int ZONE;

	/* Pointers to pdf and cdf integrand functions */
	double *(*stable_v_pdf_g)(double, void*, double*, size_t, unsigned int, unsigned int*);
	double *(*stable_v_cdf_g)(double, void*, double*, size_t, unsigned int, unsigned int*);

	/* Precalculated values */
	double alfainvalfa1;     /* alfa/(alfa-1)*/
	double xi;               /* -beta*tan(alfa*pi/2)*/
	double theta0;           /* 1/alfa*atan(beta*(tan(alfa*pi/2))=atan(-xi)/alfa;*/
	double c1, c2_part, c3;  /* additive and multiplicative constants*/
	double k1;               /* cos(alfa*theta0)^(1/(alfa-1)) = (1+xi^2)^(-0.5/(alfa-1));*/
	double S;                /* (1+xi^2)^(1/(2*alfa));*/
	double Vbeta1;           /*pow(1/dist->alfa,dist->alfainvalfa1) *
                                   (dist->alfa-1)*pow(-cos(dist->alfa*PI_2),1/(dist->alfa-1))*/

	/* These ones change from point to point of evaluation */
	double theta0_; /* theta0_ = +-theta0 */
	double beta_;
	double *xxipow;  /* (x-xi)^(alfa/(alfa-1))*/

	gsl_integration_workspace * gslworkspace;
	gsl_rng * gslrand;
}
StableDistV;

typedef struct {
	double (*ptr_funcion)(StableDist *dist, const double x, double *err);
	StableDist *dist;
	const double *x;
	int Nx;
	double *pdf;
	double *err;
}
StableArgsPdf;

typedef struct {
	double (*ptr_funcion)(StableDist *dist, const double x, double *err);
	StableDist *dist;
	const double *x;
	int Nx;
	double *cdf;
	double *err;
}
StableArgsCdf;

unsigned int stable_v_get_THREADS();
void   stable_v_set_THREADS(unsigned int threads);

unsigned int stable_v_get_SUBS();
void   stable_v_set_SUBS(unsigned int subs);

unsigned int stable_v_get_IT_MAX();
void   stable_v_set_IT_MAX(unsigned int itmax);

int stable_v_get_METHOD();
int stable_v_get_METHODNAME(char *s);
void stable_v_set_METHOD(int method);

FILE * stable_v_get_file_LOG();
void stable_v_set_file_LOG(FILE * flog, char *name);

double stable_v_get_absTOL();
void   stable_v_set_absTOL(double abstol);

double stable_v_get_relTOL();
void   stable_v_set_relTOL(double reltol);

double stable_v_get_ALFA_TH();
void   stable_v_set_ALFA_TH(double alfath);

double stable_v_get_BETA_TH();
void   stable_v_set_BETA_TH(double betath);

double stable_v_get_XXI_TH();
void   stable_v_set_XXI_TH(double xxith);

double stable_v_get_THETA_TH();
void   stable_v_set_THETA_TH(double thetath);

FILE * stable_v_get_FINTEG();
FILE * stable_v_set_FINTEG(char * filename);

FILE * stable_v_get_FLOG();
FILE * stable_v_set_FLOG(char * filename);

StableDistV *stable_v_create(double alfa, double beta, double sigma, double mu,
							 int parametrization);

StableDistV *stable_v_copy(StableDistV *src_dist);

void stable_v_free(StableDistV *dist);

int stable_v_setparams(StableDistV *dist,
					   double alfa, double beta, double sigma, double mu,
					   int parametrization);

int stable_v_checkparams(double alfa, double beta, double sigma, double mu,
						 int parametrization);

double stable_v_pdf_point(StableDistV *dist, const double x, double *err);

double stable_v_cdf_point(StableDistV *dist, const double x, double *err);

double stable_v_rnd_value(StableDistV *dist);

void stable_v_pdf(StableDistV *dist, const double x[], const unsigned int Nx,
				  double *pdf, double *err);

void stable_v_cdf(StableDistV *dist, const double x[], const unsigned int Nx,
				  double *cdf, double *err);

double *stable_v_rnd(StableDistV *dist, const unsigned int n);

void stable_v_integration(StableDistV *dist, double(func)(double, void*),
						  double a, double b, double epsabs, double epsrel,
						  unsigned short limit,
						  double *result, double *abserr, unsigned short method);
//void stable_v_error_handler(int errnum);


/************************************************************************
 ************************************************************************
 * Parameter estimation                                                 *
 ************************************************************************
 ************************************************************************/


/******************************************************************************/
/*        Parameter estimation structure                                      */
/******************************************************************************/
typedef struct {
	StableDist *dist;
	double *data;
	double *pdf;
	double *err;
	unsigned int length;
	double nu_c;
	double nu_z;
}
stable_like_params;

/* Estimation functions */

short stable_fit_init(StableDist *dist, const double *data,
					  const unsigned int length,  double *nu_c, double *nu_z);

int stable_fit_koutrouvelis(StableDist *dist, const double *data, const unsigned int length);

int stable_fit(StableDist *dist, const double *data, const unsigned int length);

int stable_fit_mle(StableDist *dist, const double *data, const unsigned int length);

int stable_fit_mle2d(StableDist *dist, const double *data, const unsigned int length);

int stable_fit_whole(StableDist *dist, const double *data, const unsigned int length);

int stable_fit_mixture(StableDist *dist, const double* data, const unsigned int length);

/* Auxiliary functions */

gsl_complex stable_samplecharfunc_point(const double x[],
										const unsigned int N, double t);

void stable_samplecharfunc(const double x[], const unsigned int Nx,
						   const double t[], const unsigned int Nt, gsl_complex * z);

void stable_fft(double *data, const unsigned int length, double * y);

double stable_loglikelihood(StableDist *dist, double *data, const unsigned int length);

//stable_like_params

int stable_fit_iter_whole(StableDist *dist, const double * data, const unsigned int length);

int stable_fit_iter(StableDist *dist, const double * data,
					const unsigned int length, const double nu_c, const double nu_z);

double stable_loglike_p(stable_like_params *params);

double stable_minusloglikelihood(const gsl_vector * theta, void * p);

/************************************************************************
 ************************************************************************
 * Random numbers generation                                            *
 ************************************************************************
 ************************************************************************/
short stable_rnd(StableDist *dist, double*rnd, const unsigned int n);
short stable_rnd_gpu(StableDist *dist, double*rnd, const unsigned int n);

double stable_rnd_point(StableDist *dist);

void stable_rnd_seed(StableDist * dist, unsigned long int s);

/************************************************************************
 ************************************************************************
 * Mixtures
 ************************************************************************
 ************************************************************************/

/**
 * @internal
 * A base function for evaluating mixtures.
 *
 * @param dist    Base distribution.
 * @param x       Array with the points to be evaluated.
 * @param Nx      Number of points in the array.
 * @param result1 Result 1.
 * @param result2 Result 2.
 * @param eval    Evaluation function (e.g, stable_pdf, stable_cdf...)
 */
void _stable_evaluate_mixture(StableDist *dist, const double x[], const int Nx,
							  double *result1, double *result2,
							  array_evaluator eval);


#endif //STABLE_API_H
