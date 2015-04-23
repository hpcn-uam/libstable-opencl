/* stable/stable_fit.c
 *
 * Functions employed by different methods of estimation implemented
 * in Libstable.
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
#include "mcculloch.h"

#include <gsl/gsl_complex.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_erf.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_fft_real.h>

#define ESTM_2D_EPSABS 0.008
#define ESTM_2D_MAX_ITER 300
#define ESTM_4D_EPSABS 0.008
#define ESTM_4D_MAX_ITER 300


void get_original(const gsl_vector *s, double *a, double *b, double *c, double *m);
void set_expanded(gsl_vector *s, const double a, const double b, const double c, const double m);

void stable_fft(double *data, const unsigned int length, double * y)
{
	//int i;

	memcpy ( (void *)y, (const void *) data, length * sizeof(double));

	gsl_fft_real_radix2_transform (y, 1, length);

	return;
}

double stable_loglikelihood(StableDist *dist, double *data, const unsigned int length)
{
	double *pdf = NULL;
	double l = 0.0;
	int i;

	pdf = malloc(sizeof(double) * length);

	stable_pdf(dist, data, length, pdf, NULL);

	for (i = 0; i < length; i++)
	{
		if (pdf[i] > 0.0) l += log(pdf[i]);
	}

	free(pdf);
	return l;
}

double stable_loglike_p(stable_like_params *params)
{
	double l = 0.0;
	int i;

	if (params->dist->gpu_enabled)
		stable_pdf_gpu(params->dist, params->data, params->length, params->pdf, NULL);
	else
		stable_pdf(params->dist, params->data, params->length, params->pdf, NULL);

	for (i = 0; i < params->length; i++)
		if (params->pdf[i] > 0.0)
			l += log(params->pdf[i]);

	return l;
}

double stable_minusloglikelihood(const gsl_vector * theta, void * p)
{
	/*Esta es la funcion a minimizar, con la estimacion de sigma y mu en cada iter*/
	double alfa = 1, beta = 0, sigma = 1.0, mu = 0.0;
	double minusloglike = 0;
	stable_like_params * params = (stable_like_params *) p;

	alfa = gsl_vector_get(theta, 0);
	beta = gsl_vector_get(theta, 1);

	/*Estima sigma y mu con McCulloch. Necesita los estadisticos nu_c nu_z*/
	czab(alfa, beta, params->nu_c, params->nu_z, &sigma, &mu);

	/*Para que la estimacion no se salga del espacio de parametros*/
	if (stable_setparams(params->dist, alfa, beta, sigma, mu, 0) < 0)
	{
		return GSL_NAN;
	}
	else minusloglike = -stable_loglike_p(params);

	if (isinf(minusloglike) || isnan(minusloglike)) minusloglike = GSL_NAN;

	return minusloglike;
}

int compare (const void * a, const void * b)
{
	/* Relacion de orden para qsort*/
	//double d=*(double *)a - *(double *)b;
	//if(d<0) return -1;
	//if(d>0) return 1;
	//else return 0;

	return ((*(double *)b < * (double *)a) - (*(double *)a < * (double *)b));
}

inline void get_original(const gsl_vector *s, double *a, double *b, double *c, double *m)
{
	*a = M_2_PI * atan(gsl_vector_get(s, 0)) + 1.0;
	*b = M_2_PI * atan(gsl_vector_get(s, 1));
	*c = exp(gsl_vector_get(s, 2));
	*m = gsl_vector_get(s, 3);
}

inline void set_expanded(gsl_vector *s, const double a, const double b, const double c, const double m)
{
	gsl_vector_set(s, 0, tan(M_PI_2 * (a - 1.0)));
	gsl_vector_set(s, 1, tan(M_PI_2 * b));
	gsl_vector_set(s, 2, log(c));
	gsl_vector_set(s, 3, m);
}

double stable_minusloglikelihood_whole(const gsl_vector * theta, void * p)
{
	/*Esta es la funcion a minimizar, en 4D*/
	double alfa = 1, beta = 0, sigma = 1.0, mu = 0.0;
	double minusloglike = 0;
	stable_like_params * params = (stable_like_params *) p;

	get_original(theta, &alfa, &beta, &sigma, &mu);

	/*Para que la estimacion no se salga del espacio de parametros*/
	if (stable_setparams(params->dist, alfa, beta, sigma, mu, 0) < 0)
	{
		printf("setparams error: %f %f %f %f\n", alfa, beta, sigma, mu);
		return GSL_NAN;
	}
	else minusloglike = -stable_loglike_p(params);

	if (isinf(minusloglike) || isnan(minusloglike)) minusloglike = GSL_NAN;

//  printf("minusloglikelihood_whole: %f\n", minusloglike);
	return minusloglike;
}

//stable_like_params
short stable_fit_init(StableDist *dist, const double * data, const unsigned int length, double *pnu_c, double *pnu_z)
{
	double *sorted = NULL;
	double alfa0, beta0, sigma0, mu1;
	//int c;
	//stable_like_params p;


	sorted = malloc(length * sizeof(double));

	memcpy ( (void *)sorted, (const void *) data, length * sizeof(double));
	qsort  ( sorted, length, sizeof(double), compare);

	//estimar con mcculloch para inicializar
	stab((const double *) sorted, length, 0, &alfa0, &beta0, &sigma0, &mu1);

	//punto inicial se mete en la dist
	if (stable_setparams(dist, alfa0, beta0, sigma0, mu1, 0) < 0)
	{
		printf("INITIAL ESTIMATED PARAMETER ARE NOT VALID");
		fflush(stdout);
		return -1;
	}

	//necesarios los estadisticos para estimar sigma y mu en cada iteracion
	if(pnu_c != NULL && pnu_z !=NULL)
		cztab(sorted, length, pnu_c, pnu_z);

	free(sorted);
	return 0;
}

int stable_fit_iter(StableDist *dist, const double * data, const unsigned int length, const double nu_c, const double nu_z)
{
	const gsl_multimin_fminimizer_type *T;
	gsl_multimin_fminimizer *s;

	gsl_multimin_function likelihood_func;

	gsl_vector *theta, *ss;

	unsigned int iter = 0;
	int status = 0;
	double size = 0;

	double a = 1, b = 0.0, c = 1, m = 0.0;
	stable_like_params par;

	par.dist = dist;
	par.data = (double *)data;
	par.length = length;
	par.nu_c = nu_c;
	par.nu_z = nu_z;
	par.pdf = (double*) calloc(length, sizeof(double));
	par.err = (double*) calloc(length, sizeof(double));

	/* Inicio: Debe haberse inicializado dist con alfa y beta de McCulloch */
	theta = gsl_vector_alloc(2);
	gsl_vector_set (theta, 0, dist->alfa);
	gsl_vector_set (theta, 1, dist->beta);

#ifdef DEBUG
	printf("%lf, %lf\n", gsl_vector_get (theta, 0), gsl_vector_get (theta, 1));
#endif

	/* Saltos iniciales */
	ss = gsl_vector_alloc (2);
	gsl_vector_set_all (ss, 0.01);

	/* Funcion a minimizar */
	likelihood_func.n = 2; // Dimension 2 (alfa y beta)
	likelihood_func.f = &stable_minusloglikelihood;
	likelihood_func.params = (void *) (&par);  // Parametros de la funcion

	/* Creacion del minimizer */
	T = gsl_multimin_fminimizer_nmsimplex2rand;

	s = gsl_multimin_fminimizer_alloc (T, 2); /* Dimension 2*/

	/* Poner funcion, estimacion inicial, saltos iniciales */
	gsl_multimin_fminimizer_set (s, &likelihood_func, theta, ss);

#ifdef DEBUG
	printf("5\n");
#endif

	/* Iterar */
	do
	{
		iter++;
		status = gsl_multimin_fminimizer_iterate(s);
//     if (status!=GSL_SUCCESS) {
//       printf("Minimizer warning: %s\n",gsl_strerror(status));
//       fflush(stdout);
//      }

		size = gsl_multimin_fminimizer_size (s);
		status = gsl_multimin_test_size (size, ESTM_2D_EPSABS);
		/*
					if (status == GSL_SUCCESS)
						{
							printf ("              converged to minimum at\n");
						}

					printf ("%5d %1.5f %1.5f %1.5f %1.5f f() = %1.8e size = %.5f\n",
									(int)iter,
									gsl_vector_get (s->x, 0),
									gsl_vector_get (s->x, 1),
									p->dist->sigma,
									p->dist->mu_1,
									s->fval, size);
						//}
		*/
	} while (status == GSL_CONTINUE && iter < ESTM_2D_MAX_ITER);

//  if (status!=GSL_SUCCESS)
//    {
//      printf("Minimizer warning: %s\n",gsl_strerror(status));
//      fflush(stdout);
//    }

	/* Se recupera la estimacion alfa y beta */
	gsl_vector_free(theta);
	/*
		theta = gsl_multimin_fminimizer_x (s);
		a = gsl_vector_get (theta, 0);
		b = gsl_vector_get (theta, 1);
	*/
	a = gsl_vector_get (s->x, 0);
	b = gsl_vector_get (s->x, 1);

	/* Y se estima sigma y mu para esos alfa y beta */
	czab(a, b, nu_c, nu_z, &c, &m);

	//printf("%5d %10.3e %10.3e %10.3e %10.3e\n",(int)iter,a,b,c,m);

	// Se almacena el punto estimado en la distribucion, comprobando que es valido
	if (stable_setparams(dist, a, b, c, m, 0) < 0)
	{
		printf("FINAL ESTIMATED PARAMETER ARE NOT VALID\n  a = %f  b = %fn  c = %f  m = %f\n", a, b, c, m);
		fflush(stdout);
	}

	gsl_vector_free(ss);
	gsl_multimin_fminimizer_free (s);
	free(par.err);
	free(par.pdf);

	return status;
}

int stable_fit(StableDist *dist, const double *data, const unsigned int length)
{
	double nu_c = 0.0, nu_z = 0.0;
	int status = 0;

	stable_fit_init(dist, data, length, &nu_c, &nu_z);
	status = stable_fit_iter(dist, data, length, nu_c, nu_z);

	return status;
}

int stable_fit_iter_whole(StableDist *dist, const double * data, const unsigned int length)
{
	const gsl_multimin_fminimizer_type *T;
	gsl_multimin_fminimizer *s;

	gsl_multimin_function likelihood_func;

	gsl_vector *theta, *ss;

	unsigned int iter = 0;
	int status = 0;
	double size = 0;

	double a = 1, b = 0.0, c = 1, m = 0.0;
	stable_like_params par;

	par.dist = dist;
	par.data = (double *)data;
	par.length = length;
	par.nu_c = 0;
	par.nu_z = 0;
	par.pdf = (double*) calloc(length, sizeof(double));
	par.err = (double*) calloc(length, sizeof(double));

	/* Inicio: Debe haberse inicializado dist con McCulloch */
	theta = gsl_vector_alloc(4);
	set_expanded(theta, dist->alfa, dist->beta, dist->sigma, dist->mu_1);

#ifdef DEBUG
	printf("%lf, %lf, %lf, %lf\n", gsl_vector_get (theta, 0), gsl_vector_get (theta, 1), gsl_vector_get (theta, 2), gsl_vector_get (theta, 3));
#endif

	/* Saltos iniciales */
	ss = gsl_vector_alloc (4);
	gsl_vector_set_all (ss, 0.01);

	/* Funcion a minimizar */
	likelihood_func.n = 4; // Dimension 4
	likelihood_func.f = &stable_minusloglikelihood_whole;
	likelihood_func.params = (void *) (&par);  // Parametros de la funcion

	/* Creacion del minimizer */
	T = gsl_multimin_fminimizer_nmsimplex2rand;

	s = gsl_multimin_fminimizer_alloc (T, 4); /* Dimension 4 */

	/* Poner funcion, estimacion inicial, saltos iniciales */
	gsl_multimin_fminimizer_set (s, &likelihood_func, theta, ss);


#ifdef DEBUG
	printf("5\n");
#endif

	/* Iterar */
	do
	{
		iter++;
		status = gsl_multimin_fminimizer_iterate(s);
		if (status != GSL_SUCCESS) {
			printf("Minimizer warning: %s\n", gsl_strerror(status));
			fflush(stdout);
		}

		size   = gsl_multimin_fminimizer_size (s);
		status = gsl_multimin_test_size (size, ESTM_4D_EPSABS);

//      printf(" %03d\t size = %f a_ = %f  b_ = %f  c_ = %f  m_ = %f\n",iter,size,gsl_vector_get (s->x, 0),gsl_vector_get (s->x, 1),
//                                                           gsl_vector_get (s->x, 2),gsl_vector_get (s->x, 3));
//      fflush(stdout);

	} while (status == GSL_CONTINUE && iter < ESTM_4D_MAX_ITER);



	if (status != GSL_SUCCESS)
	{
		printf("Minimizer warning: %s\n", gsl_strerror(status));
		fflush(stdout);
	}

	/* Se recupera la estimacion */

	gsl_vector_free(theta);
	theta = gsl_multimin_fminimizer_x (s);
	get_original(theta, &a, &b, &c, &m);

	// Se almacena el punto estimado en la distribucion, comprobando que es valido
	if (stable_setparams(dist, a, b, c, m, 0) < 0)
	{
		printf("FINAL ESTIMATED PARAMETER ARE NOT VALID\n  a = %f  b = %fn  c = %f  m = %f\n", a, b, c, m);
		fflush(stdout);
	}

	gsl_vector_free(ss);
	gsl_multimin_fminimizer_free (s);
	free(par.err);
	free(par.pdf);

	return status;
}

int stable_fit_whole(StableDist *dist, const double *data, const unsigned int length)
{
//  double nu_c=0.0,nu_z=0.0;
	int status = 0;
//  stable_fit_init(dist,data,length,&nu_c,&nu_z);
//  printf("McCulloch %d muestras: %f %f %f %f\n",length,dist->alfa,dist->beta,dist->sigma,dist->mu_1);

	status = stable_fit_iter_whole(dist, data, length);

	return status;
}



double * load_rand_data(char * filename, int N)
{
	FILE * f_data;
	double * data;
	int i;

	if ((f_data = fopen(filename, "rt")) == NULL)
	{
		perror("Error when opening file with random data");
	}

	data = malloc(N * sizeof(double));

	for (i = 0; i < N; i++)
	{
		if (EOF == fscanf(f_data, "%le\n", data + i))
		{
			perror("Error when reading data");
		}
	}
	return data;
}

int stable_fit_mle(StableDist *dist, const double *data, const unsigned int length) {
	return stable_fit_whole(dist, data, length);
}

int stable_fit_mle2d(StableDist *dist, const double *data, const unsigned int length) {
	return stable_fit(dist, data, length);
}

