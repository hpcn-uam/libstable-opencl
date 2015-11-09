/* stable/stable_inv.h
 *
 * Functions to calculate the quantile function of alpha-stable
 * distributions. Based on the developed method for CDF evaluation.
 * Code fractions based on code in [1].
 *
 * [1] Mark S. Veillete. Alpha-Stable Distributions in MATLAB
 *     http://math.bu.edu/people/mveillet/research.html
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
#include "stable_integration.h"
#include <gsl/gsl_roots.h>
#include <gsl/gsl_cdf.h>

#include "methods.h"
#include <pthread.h>
#include "stable_inv_precalcs.h"

double stable_quick_inv_point(StableDist *dist, const double q, double *err)
{
  double x0 = 0;
  double C = 0;
  double alfa = dist->alfa;
  double beta = dist->beta;
  double q_ = q;
  double signBeta = 1;

  if (alfa<0.1)   alfa=0.1;

  if (beta < 0) {
    signBeta = -1;
    q_ = 1.0 - q_;
    beta = -beta;
  }
  if (beta==1) {
    if (q_<0.1) {
      q_=0.1;
    }
  }

  /* Asympthotic expansion near the limits of the domain */
  if (q_>0.9 || q_<0.1) {
    if (alfa!=1.0)
      C = (1-alfa)/(exp(gammaln(2-alfa))*cos(M_PI*alfa/2.0));
    else
      C = 2/M_PI;

    if (q_>0.9)
      x0=pow((1-q_)/(C*0.5*(1.0+beta)),-1.0/alfa);
    else
      x0=-pow(q_/(C*0.5*(1.0-beta)),-1.0/alfa);

    *err = 0.1;
  }

  else {
	/* Linear interpolation on precalculated values */
    int ia, ib, iq;
    double aux=0;
    double xa = modf(alfa/0.1,&aux); ia =(int)aux-1;
    double xb = modf(beta/0.2,&aux); ib =(int)aux;
    double xq = modf(  q_/0.1,&aux); iq =(int)aux-1;

    if (alfa==2) {ia = 18; xa = 1.0;}
    if (beta==1) {ib = 4;  xb = 1.0;}
    if (q_==0.9) {iq = 7;  xq = 1.0;}

    double p[8] = {precalc[iq][ib][ia],   precalc[iq][ib][ia+1],   precalc[iq][ib+1][ia],   precalc[iq][ib+1][ia+1],
                   precalc[iq+1][ib][ia], precalc[iq+1][ib][ia+1], precalc[iq+1][ib+1][ia], precalc[iq+1][ib+1][ia+1]};

	//Trilinear interpolation
    x0=((p[0]*(1.0-xa)+p[1]*xa)*(1-xb)+(p[2]*(1-xa)+p[3]*xa)*xb)*(1-xq)+((p[4]*(1.0-xa)+p[5]*xa)*(1-xb)+(p[6]*(1-xa)+p[7]*xa)*xb)*xq;

    if (err!=NULL) {
      *err = fabs(0.5*(p[0]-p[1]));
    }

#ifdef DEBUG
    printf("Quickinv: %f %f %f %d %d %d %f %f %f\n",alfa,beta,q_,ia,ib,iq,xa,xb,xq);
    printf("Precalc: \t %1.6e %1.6e \t\t %1.6e %1.6e\n\t\t %1.6e %1.6e \t\t %1.6e %1.6e",p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]);
#endif
  }

  x0=x0*signBeta*dist->sigma + dist->mu_0;

  return x0;
}

typedef struct {
  StableDist *dist;
  double q;
} rootparams;

double f_wrap(double x, void * params) {
  rootparams *par = (rootparams *)params;
  double y = stable_cdf_point(par->dist,x,NULL)-par->q;
#ifdef DEBUG
  printf ("F(%e)=%e\n",x,y);
#endif
  return y;
}
double df_wrap(double x, void * params) {
  rootparams *par = (rootparams *)params;
  double dy = stable_pdf_point(par->dist,x,NULL);
#ifdef DEBUG
  printf ("dF(%e)=%e\n",x,dy);
#endif
  return dy;
}
void fdf_wrap(double x, void * params, double * f, double * df) {
  rootparams *par = (rootparams *)params;
  *f  = stable_cdf_point(par->dist,x,NULL)-par->q;
  *df = stable_pdf_point(par->dist,x,NULL);
#ifdef DEBUG
  printf ("Fdf(%e)=(%e,%e)\n",x,*f,*df);
#endif
  return;
}

static int _dbl_compare (const void * a, const void * b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;

    return (db < da) - (da < db);
}

static short _is_guess_valid(double val)
{
    return fabs(val) < 1e-6;
}

static double _stable_inv_bracket(StableDist* dist, double q, size_t point_count, double guess, double search_width, double* guess_error)
{
    double points[point_count];
    double cdf[point_count];
    double interval_begin, interval_end, interval_step;
    size_t i, bracket_begin, bracket_end;
    double bracket_begin_val, bracket_end_val;

    interval_begin = guess - search_width / 2;
    interval_end = guess + search_width / 2;
    interval_step = search_width / point_count;

    for(i = 0; i < point_count; i++)
        points[i] = interval_begin + interval_step * i;

    stable_cdf_gpu(dist, points, point_count, cdf, NULL);

    // Binary search for the 0
    bracket_begin = 0;
    bracket_end = point_count - 1;

    bracket_begin_val = cdf[bracket_begin] - q;
    bracket_end_val = cdf[bracket_end] - q;

    if(_is_guess_valid(bracket_begin_val))
    {
        guess_error = 0;
        return points[bracket_begin];
    }
    else if(_is_guess_valid(bracket_end_val))
    {
        guess_error = 0;
        return points[bracket_end];
    }

    while(bracket_end - bracket_begin > 1)
    {
        bracket_begin_val = cdf[bracket_begin] - q;
        bracket_end_val = cdf[bracket_end] - q;

        if(bracket_begin_val > 0)
        {
            *guess_error = search_width;
            return points[bracket_begin];
        }
        else if(bracket_end_val < 0)
        {
            *guess_error = search_width;
            return points[bracket_end];
        }

        size_t middle = (bracket_end + bracket_begin + 1) / 2;
        double middle_val = cdf[middle] - q;

        if(_is_guess_valid(middle_val))
        {
            *guess_error = 0;
            return points[middle];
        }

        if(middle_val < 0)
            bracket_begin = middle;
        else
            bracket_end = middle;
    }

    *guess_error = interval_step / 2;

    return (points[bracket_end] + points[bracket_begin]) / 2;
}

double stable_inv_point_gpu(StableDist* dist, const double q, double *err)
{
    double guess;
    double interval_width;
    double tolerance = 1e-4;
    double guess_error;
    size_t point_count = 10;

    if(dist->ZONE == GAUSS || dist->ZONE == CAUCHY || dist->ZONE == LEVY)
        return stable_inv_point(dist, q, err);

    guess = stable_quick_inv_point(dist, q, &guess_error);

    if(q > 0.9 || q < 0.1)
        return guess;

    while(guess_error > tolerance)
    {
        interval_width = 2 * guess_error;
        guess = _stable_inv_bracket(dist, q, point_count, guess, interval_width, &guess_error);
    }

    if(err)
        *err = guess_error;

    return guess;
}

double stable_inv_point(StableDist *dist, const double q, double *err)
{
  double x,x0=0;

  gsl_root_fdfsolver * fdfsolver;

  // Casos particulares
  if (dist->ZONE == GAUSS) {
    x = gsl_cdf_ugaussian_Pinv(q)*M_SQRT2*dist->sigma+dist->mu_0;
    return x;
  }
  else if (dist->ZONE == CAUCHY) {
    x = tan(M_PI*(q - 0.5))*dist->sigma+dist->mu_0;
    return x;
  }
  else if (dist->ZONE == LEVY) {
    x = (dist->beta*pow(gsl_cdf_ugaussian_Pinv(q/2.0),-2.0)+dist->xi)*dist->sigma+dist->mu_0;
    return x;
  }

  x = stable_quick_inv_point(dist, q, err);

  rootparams params;
  params.dist = dist;
  params.q = q;

  gsl_function_fdf fdf;
  fdf.f   = f_wrap;
  fdf.df  = df_wrap;
  fdf.fdf = fdf_wrap;
  fdf.params = (void*)&params;

//  gsl_function f;
//  f.function = f_wrap;
//  f.params = (void*)&params;

  int status;
  if (INV_MAXITER>0) {
    fdfsolver = gsl_root_fdfsolver_alloc (gsl_root_fdfsolver_secant);
    gsl_root_fdfsolver_set (fdfsolver, & fdf, x);

//    fsolver = gsl_root_fsolver_alloc(gsl_root_fsolver_falsepos);
//    gsl_root_fsolver_set (fsolver, & f, x_lower , x_upper);

    int k=0;
	double INVrelTOL = 1e-6;
    do {
      k++;
      status = gsl_root_fdfsolver_iterate (fdfsolver);
      x0 = x;
      x = gsl_root_fdfsolver_root (fdfsolver);
      status = gsl_root_test_delta (x, x0, 0, INVrelTOL);
    } while (status == GSL_CONTINUE && k < INV_MAXITER);

    gsl_root_fdfsolver_free(fdfsolver);
  }

#ifdef DEBUG
  if (status == GSL_SUCCESS) {
    printf("convergence at x = %e\n",x);
  }
  else {
    printf("didn't converge\n");
  }
#endif

  return x;
}


void * thread_init_inv(void *ptr_args)
{
  StableArgsCdf *args = (StableArgsCdf *)ptr_args;
  int counter_ = 0;

  while (counter_ < args->Nx)
    {
      args->cdf[counter_]=(*(args->ptr_funcion))(args->dist,args->x[counter_],
                                                 &(args->err[counter_]));
      counter_++;
    }
  pthread_exit(NULL);
}

void stable_inv(StableDist *dist, const double q[], const int Nq,
                double *inv, double *err)
{
  int Nq_thread[THREADS],
      initpoint[THREADS],
      k,flag=0;
  void *status;
  pthread_t threads[THREADS];
  StableArgsCdf args[THREADS];

  /* If no error pointer is introduced, it's created*/
  if (err==NULL) {flag=1;err=malloc(Nq*sizeof(double));}

  /* Evaluation points divided among available threads */
  /* Reparto de los puntos de evaluacion entre los hilos disponibles */
  Nq_thread[0] = Nq/THREADS;
  if (0 < Nq%THREADS) Nq_thread[0]++;

  initpoint[0] = 0;
  for(k=1;k<THREADS;k++)
    {
      Nq_thread[k] = Nq/THREADS;
      if (k < Nq%THREADS) Nq_thread[k]++;
      initpoint[k] = initpoint[k-1] + Nq_thread[k-1];
    }

  /* Threads' creation. A copy of the ditribution is created for each of them*/
  /* Creacion de los hilos, pasando a cada uno una copia de la distribucion */
  for(k=0; k<THREADS; k++)
    {
      args[k].ptr_funcion = stable_inv_point;

      args[k].dist = stable_copy(dist);
      args[k].cdf  = inv+initpoint[k];
      args[k].x    = q+initpoint[k];
      args[k].Nx   = Nq_thread[k];
      args[k].err  = err+initpoint[k];

      if(pthread_create(&threads[k], NULL, thread_init_inv, (void *)&args[k]))
        {
          perror("Error en la creacion de hilo");
          if (flag==1) free(err);
          return;
        }
    }

  /* Wait until treadhs execution terminates */
  /* Esperar a finalizacion de todos los hilos */
  for(k=0; k<THREADS; k++)
    {
      pthread_join(threads[k], &status);
    }

  /* Free distribution copies */
  /* Liberar las copias de la distribucion realizadas */
  for(k=0; k<THREADS; k++)
    {
      stable_free(args[k].dist);
    }

  if (flag==1) free(err);
}

short stable_inv_gpu(StableDist *dist, const double q[], const int Nq,
                double *inv, double *err)
{
    if(dist->ZONE == GAUSS || dist->ZONE == CAUCHY || dist->ZONE == LEVY)
    {
      stable_inv(dist, q, Nq, inv, err); // Rely on analytical formulae where possible
      return 0;
    }
    else
    {
        stable_clinteg_set_mode(&dist->cli, mode_quantile);
        if(dist->gpu_queues == 1)
            return stable_clinteg_points(&dist->cli, (double*) q, inv, NULL, err, Nq, dist);
        else
            return stable_clinteg_points_parallel(&dist->cli, (double*) q, inv, NULL, err, Nq, dist, dist->gpu_queues);
    }
}
