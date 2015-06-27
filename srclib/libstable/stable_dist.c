/* stable/stable_dist.c
 *
 * Main Libstable source file. Definition of the StableDist structures
 * and auxiliary functions to manage alpha-stable distributions.
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
#include <gsl/gsl_errno.h>

#include "stable_api.h"
//#include "stable_common.h"

#include <pthread.h>
#include <math.h>
#include <unistd.h>

/*----------------------------------------------------------------------------*/
/*                             Public part                                    */
/*----------------------------------------------------------------------------*/

unsigned int stable_get_THREADS() { return THREADS; }
void stable_set_THREADS(unsigned int value)
{
  if (value <= 0) THREADS = sysconf(_SC_NPROCESSORS_ONLN);
  else THREADS = value;
  //printf("\nCPUs = %u\n",THREADS);
}

int stable_get_METHOD() { return METHOD; }
void stable_set_METHOD(int value) { METHOD = value; }
int stable_get_METHOD2() { return METHOD2; }
void stable_set_METHOD2(int value) { METHOD2 = value; }
int stable_get_METHOD3() { return METHOD3; }
void stable_set_METHOD3(int value) { METHOD3 = value; }

unsigned int stable_get_IT_MAX() { return IT_MAX; }
void stable_set_IT_MAX(unsigned int value) { IT_MAX = value;}

unsigned int stable_get_INV_MAXITER() { return INV_MAXITER; }
void stable_set_INV_MAXITER(unsigned int value) { INV_MAXITER = value;}

double stable_get_relTOL() { return relTOL; }
void stable_set_relTOL(double value)
{
  relTOL = value;
  //FACTOR = 1e-16;
  if (value<1e-11)
    METHOD_= STABLE_QAG2; //quadstep
  else
    METHOD_= STABLE_QNG;
}

double stable_get_absTOL() { return absTOL; }
void stable_set_absTOL(double value)
{
  absTOL = value;
  //FACTOR = 1e-16;
}

/* Parameter thresholds */

/* When abs(alpha - 1)<ALFA_TH alpha is set to 1 */
double stable_get_ALFA_TH() { return ALFA_TH; }
void stable_set_ALFA_TH(double value) { ALFA_TH = value; }

/* When 1-abs(beta)<BETA_TH beta is set to sign(beta)*1.0 */
/* When alpha = 1 and abs(beta)<BETA_TH beta is set to 0.0*/
double stable_get_BETA_TH() { return BETA_TH; }
void stable_set_BETA_TH(double value) { BETA_TH = value; }

/* When abs(x-xxi)<XXI_TH x is set to XXI */
double stable_get_XXI_TH() { return XXI_TH; }
void stable_set_XXI_TH(double value) { XXI_TH = value; }

/* When theta get closer than THETA_TH to integration interval limits theta is set to the limit value */
double stable_get_THETA_TH() { return THETA_TH; }
void stable_set_THETA_TH(double value) { THETA_TH = value; }

/* Debug purposes*/
FILE * stable_get_FINTEG() { return FINTEG; }
FILE * stable_set_FINTEG(char *name)
{
  FINTEG = fopen(name,"wt");
  return FINTEG;
}

/* Log-file configuration */
FILE * stable_get_FLOG() { return FLOG; }
FILE * stable_set_FLOG(char * name)
{
  FLOG = fopen(name,"wt");
  return FLOG;
}

void stable_clear_LOG()
{
  /* Not implemented yet */
  char *name;

  name = "nombre";
  fclose(FLOG);
  FLOG = fopen(name,"wt");
}

int stable_setparams(StableDist *dist,
                     double alfa, double beta, double sigma, double mu,
                     int parametrization)
{
  int zona;

  if(dist==NULL)
    {
      printf("ERROR");
      exit(2);
    }
  if((zona = stable_checkparams(alfa,beta,sigma,mu,parametrization)) == NOVALID)
    {
    //  printf ("No valid parameters: %lf %lf %lf %lf %d\n",
    //         alfa, beta, sigma, mu, parametrization);
      return zona;
    }

  dist->alfa = alfa;
  dist->beta = beta;
  dist->sigma = sigma;

  switch (zona)
    {
      case STABLE_B1:
        dist->beta = (dist->beta > 0) ? 1.0 : -1.0; // Avoid rounding errors maybe?
      case STABLE:
        dist->alfainvalfa1 = alfa/(alfa-1.0);
        dist->xi = -beta*tan(0.5*alfa*M_PI);
        dist->theta0 = atan(-dist->xi)/alfa;
        //dist->k1 = pow(1.0+dist->xi*dist->xi,-0.5/(alfa-1.0));
        dist->k1 = -0.5/(alfa-1.0)*log(1.0+dist->xi*dist->xi);
        dist->S = pow(1.0+dist->xi*dist->xi,0.5/alfa);
        dist->Vbeta1 = dist->k1 - dist->alfainvalfa1 * log(dist->alfa)
                                + log(fabs(dist->alfa-1.0));
        dist->stable_pdf_point = &stable_pdf_point_STABLE;
        dist->stable_cdf_point = &stable_cdf_point_STABLE;

        if (alfa < 1.0)
          {
            dist->c1 = 0.5-dist->theta0*M_1_PI;
            dist->c2_part = alfa/((1.0-alfa)*M_PI);
            dist->c3 = M_1_PI;
          }
        else
          {
            dist->c1 = 1.0;
            dist->c2_part = alfa/((alfa-1.0)*M_PI);
            dist->c3 = -M_1_PI;
          }

        //XXI_TH = pow(10,EXP_MAX/fabs(dist->alfainvalfa1));//REVISAR CON NOLAN...
		//XXI_TH = max(XXI_TH,10*EPS);

        if (alfa>1)
         {
          AUX1=log(log(8.5358/(relTOL))/0.9599);/*3.76;*/
          AUX2=log(relTOL);/*-40;*/
         }
        else
         {
          AUX1=log(relTOL);/*-40;*/
          AUX2=log(log(8.5358/(relTOL))/0.9599);/*3.76;*/
         }

        break;

      case ALFA_1_B1:
        dist->beta = (dist->beta > 0) ? 1.0 : -1.0;
      case ALFA_1:
        dist->alfa=1;
        dist->c2_part = 0.5/fabs(beta);
        dist->alfainvalfa1 = 0.0;
        dist->xi = 0.0;
        dist->theta0 = M_PI_2;
        dist->k1 = log(2.0*M_1_PI);
        //dist->k1 = 2.0*M_1_PI;
        dist->S = 2.0*M_1_PI;
        dist->c1 = 0.0;
        dist->c3 = M_1_PI;
        dist->Vbeta1=2.0*M_1_PI/M_E;
        dist->stable_pdf_point = &stable_pdf_point_ALFA_1;
        dist->stable_cdf_point = &stable_cdf_point_ALFA_1;
		//XXI_TH = 10*EPS;
        if (beta<0)
         {
          AUX1=log(log(8.5358/(relTOL))/0.9599);/*4;*/
          AUX2=log(relTOL);/*-25;*/
         }
        else
         {
          AUX1=log(relTOL);/*-25;*/
          AUX2=log(log(8.5358/(relTOL))/0.9599);/*4;*/
         }
        break;

      case CAUCHY:
        dist->beta=0;
        dist->alfa=1;
        dist->c2_part = 0.0;
        dist->alfainvalfa1 = 0.0;
        dist->xi = 0.0;
        dist->theta0 = M_PI_2;
        dist->k1 = log(2.0*M_1_PI);
        //dist->k1 = 2.0*M_1_PI;
        dist->S = 2.0*M_1_PI;
        dist->c1 = 0.0;
        dist->c3 = M_1_PI;
        dist->Vbeta1=2.0*M_1_PI/M_E;
        dist->stable_pdf_point = &stable_pdf_point_CAUCHY;
        dist->stable_cdf_point = &stable_cdf_point_CAUCHY;
        break;

      case GAUSS:
        dist->alfa=2;
        dist->beta=0.0;
        dist->alfainvalfa1 = 2.0;
        dist->xi = 0.0;
        dist->theta0 = 0.0;
        dist->k1 = log(2.0);
        //dist->k1 = 2.0;
        dist->S = 2.0;
        dist->c1 = 1.0;
        dist->c2_part = 2.0*M_1_PI;
        dist->c3 = -M_1_PI;
        dist->Vbeta1=0.25;
        dist->stable_pdf_point = &stable_pdf_point_GAUSS;
        dist->stable_cdf_point = &stable_cdf_point_GAUSS;
        break;

      case LEVY:
        dist->alfa = 0.5;
        dist->beta = (2.0*(beta>0)-1.0);//será 1 ó -1
        dist->alfainvalfa1 = -1.0;
        dist->xi = -dist->beta;
        dist->theta0 = 0.5*M_PI;
        dist->k1 = 0.0;
        dist->S = 1.0;
        dist->c1 = 0.0;
        dist->c2_part = 0.5*M_1_PI;
        dist->c3 = M_1_PI;
        dist->Vbeta1 = dist->k1 - dist->alfainvalfa1 * log(dist->alfa)
                                + log(fabs(dist->alfa-1.0));
        dist->stable_pdf_point = &stable_pdf_point_LEVY;
        dist->stable_cdf_point = &stable_cdf_point_LEVY;
        break;
    }

  if (parametrization == 0)
    {
      dist->mu_0 = mu;
      if (dist->alfa==1)
        dist->mu_1 = mu-dist->beta*2*M_1_PI*dist->sigma*log(dist->sigma);
      else dist->mu_1 = mu+dist->xi*dist->sigma;
    }
  else if (parametrization == 1)
    {
      dist->mu_1 = mu;
      if (dist->alfa==1)
        dist->mu_0 = mu+dist->beta*2*M_1_PI*dist->sigma*log(dist->sigma);
      else dist->mu_0 = mu-dist->xi*dist->sigma;
    }

  dist->theta0_ = dist->theta0;
  dist->beta_ = dist->beta;
  dist->xxipow = 0.0;
  dist->ZONE = zona;

  return zona;
}

int stable_checkparams(double alfa, double beta, double sigma, double mu,
                       int parametrization)
{
  /*Check parameters*/
  if (0.0 >= alfa || alfa > 2.0)
    {
      //printf("Alpha must lie between 0.0 and 2.0.");
      return NOVALID;
    }
  else if (beta < -1.0 || beta > 1.0)
    {
      //printf("Beta must lie between -1.0 and 1.0.");
      return NOVALID;
    }
  else if (sigma <= 0.0)
    {
      //printf("Sigma must be positive.");
      return NOVALID;
    }
  else if (isnan(mu) || isinf(mu))
    {
      //printf("Mu must be real.");
      return NOVALID;
    }
  else if (parametrization != 0 && parametrization != 1)
    {
      //printf("Only parametrizations 0 and 1 are accepted.");
      return NOVALID;
    }

  /*ZONE determination*/
  if ((2.0 - alfa) <= ALFA_TH)
    return GAUSS;  //GAUSS
  else if (fabs(alfa-0.5) <= ALFA_TH && fabs((fabs(beta)-1.0)) <= BETA_TH)
    return LEVY; //LEVY
  else if (fabs(alfa-1.0) <= ALFA_TH && fabs(beta) <= BETA_TH)
    return CAUCHY;
  else if (fabs(alfa-1.0) <= ALFA_TH) {
//    if (fabs(fabs(beta)-1) <= BETA_TH) return ALFA_1_B1;
    return ALFA_1;
  }
  else {
//    if (fabs(fabs(beta)-1) <= BETA_TH) return STABLE_B1;
    return STABLE;
  }
  /*When alpha=1,beta=1 Landau distribution is obtained,
    but it is calculated as in the general case*/
}

StableDist * stable_create(double alfa, double beta, double sigma, double mu,
                           int parametrization)
{
  /*gsl_error_handler_t * old_handler;
  old_handler = */gsl_set_error_handler (&error_handler);

  StableDist * dist = (StableDist *) malloc(sizeof (StableDist));

  if (dist == NULL)
    {
      perror("No se pudo crear la distribucion.");
      return NULL;
    }
  if((stable_setparams(dist,alfa,beta,sigma,mu,parametrization)) == NOVALID)
    {
      perror ("No se pudo crear la distribucion.");
      return NULL;
    }
  gsl_rng_env_setup(); //leemos las variables de entorno
  dist->gslworkspace = gsl_integration_workspace_alloc(IT_MAX);
  dist->gslrand = gsl_rng_alloc (gsl_rng_default);
  dist->gpu_enabled = 0;
  dist->gpu_platform = 0;
  dist->gpu_queues = 1;

  //Allow the distribution to use THREADS threads.
  stable_set_THREADS(THREADS);

  return dist;
}

short stable_activate_gpu(StableDist* dist)
{
  if(dist->gpu_enabled)
    return 0;

  short error = stable_clinteg_init(&dist->cli, dist->gpu_platform);

  if(!error)
    dist->gpu_enabled = 1;

  return error;
}


void stable_deactivate_gpu(StableDist* dist)
{
  if(!dist->gpu_enabled)
    return;

  stable_clinteg_teardown(&dist->cli);
  dist->gpu_enabled = 0;
}

StableDist * stable_copy(StableDist *src_dist)
{
  StableDist *dist;

  dist = stable_create(src_dist->alfa, src_dist->beta,
                       src_dist->sigma, src_dist->mu_0, 0);
  return dist;
}

void stable_free(StableDist *dist)
{
  if (dist == NULL)
    return;

  if(dist->gpu_enabled)
    stable_deactivate_gpu(dist);

  gsl_integration_workspace_free(dist->gslworkspace);
  gsl_rng_free(dist->gslrand);
  free(dist);
}
