/* stable/stable_dist_vect.c
 *
 * Vectorial approach to the calculation of stable densities and
 * distribution. No paralellized code.
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
//#include "stable_common.h"
#include "methods.h"

#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_erf.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_math.h>

#include <pthread.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>

/*----------------------------------------------------------------------------*/
/*                         Private" functions                                 */
/*----------------------------------------------------------------------------*/

double stable_v_V1(double theta, void *args);
inline double stable_v_V1(double theta, void *args)
{
  StableDistV *dist = (StableDistV *)args;
  double V;

#ifdef DEBUG
  integ_eval++;
#endif
  V = (dist->beta_*theta+M_PI_2)/cos(theta);
//  V = sin(theta)*V/dist->beta_ + log(V) + dist->k1;
  V = exp(sin(theta)*V/dist->beta_)*V*dist->k1;

  return V;
}

double stable_v_V2(double theta,void *args);
inline double stable_v_V2(double theta,void *args)
{
  StableDistV *dist = (StableDistV *)args;
  double cos_theta,V;

#ifdef DEBUG
  integ_eval++;
#endif

  cos_theta = cos(theta);
  V = (dist->theta0_+theta)*dist->alfa;
//  V = log(cos_theta/sin(V))*dist->alfainvalfa1 + log(cos(V-theta)/cos_theta) + dist->k1;
  V = pow(cos_theta/sin(V),dist->alfainvalfa1)*cos(V-theta)/cos_theta*dist->k1;

  return V;
}

double *stable_v_V_vec2(const double *theta, size_t Nth, void *args, double *V)
{
  StableDistV *dist = (StableDistV *)args;
  double cos_theta[Nth];
  int l;

#ifdef DEBUG
  integ_eval+=Nth;
#endif

  for(l=0;l<Nth;l++)
    {
      cos_theta[l] = cos(theta[l]);
      V[l] = (dist->theta0_+theta[l])*dist->alfa;
//      V[l] = log(cos_theta/sin(V))*dist->alfainvalfa1 + log(cos(V-theta)/cos_theta) + dist->k1;
      V[l] = pow(cos_theta[l]/sin(V[l]),dist->alfainvalfa1)*cos(V[l]-theta[l])/cos_theta[l]*dist->k1;
    }
  return V;
}
double *stable_v_pdf_g1(double theta, void *args, double *g, size_t N, char conv[])
{
  StableDistV *dist = (StableDistV *)args;
  double /*g,*/V;
  int k;

  V = stable_v_V1(theta,args);
//  g = (double*)malloc(N*sizeof(double));
  #ifdef DEBUG
  fprintf(FINTEG,"%e\t",theta);
  #endif
  for (k=0;k<N;k++)
    {
      if (conv[k]==0.0) continue;
      g[k] = dist->xxipow[k]*V;
      g[k] = g[k]*exp(-g[k]);
      if (isnan(g[k]) || isinf(g[k]) || g[k]<0.0) g[k]=0.0;
      #ifdef DEBUG
      fprintf(FINTEG,"%e\t",g[k]);
      #endif
    }
  #ifdef DEBUG
  fprintf(FINTEG,"\n");
  #endif

  return g;
}

double *stable_v_pdf_g2_(double theta, void *args, double *g, size_t N,unsigned int first, unsigned int next[])
{
  double g_,V;
  unsigned int k;

  V = stable_v_V2(theta,args);

  k = first; //posicion en la que comienza el vector.
  while (k<N) //el final se indica con k=N;
  {
//    printf(" %d\n",k);
    g_ = ((StableDistV*)args)->xxipow[k]*V;
    g_ = g_*exp(-g_);
    if (isnan(g_) || isinf(g_) || g_<0.0) g_ = 0.0;
    g[k] = g_;
    k = next[k];
  }

  return g;
}

double *stable_v_cdf_g2(double theta, void *args, double *g, size_t N,unsigned int first, unsigned int next[])
{
  double g_,V;
  unsigned int k;

  V = stable_v_V2(theta,args);

  k = first; //posicion en la que comienza el vector.
  while (k<N) //el final se indica con k=N;
  {
//    printf(" %d\n",k);
    g_ = ((StableDistV*)args)->xxipow[k]*V;
    g_ = exp(-g_);
    if (isnan(g_) || isinf(g_) || g_<0.0) g_ = 0.0;
    g[k] = g_;
    k = next[k];
  }

  return g;
}

double *stable_v_pdf_g2(double theta, void *args, double *g, size_t N,char conv[])
{
  StableDistV *dist = (StableDistV *)args;
  double /*g,*/V;
  int k;

  V = stable_v_V2(theta,args);
//  g = (double*)malloc(N*sizeof(double));
  #ifdef DEBUG
  fprintf(FINTEG,"%e\t",theta);
  #endif
  for (k=0;k<N;k++)
    {
      if (conv[k]==1) {g[k]=0.0;}
      else {
      g[k] = dist->xxipow[k]*V;
      g[k] = g[k]*exp(-g[k]);
      if (isnan(g[k]) || isinf(g[k]) || g[k]<0.0) g[k]=0.0;
      }
      #ifdef DEBUG
      fprintf(FINTEG,"%e\t",g[k]);
      #endif
    }
  #ifdef DEBUG
  fprintf(FINTEG,"\n");
  #endif

  return g;
}

double *stable_v_pdf_g_mat2_max(const double *theta, size_t Nth, void *args, double *g, size_t *intmax, size_t Nx,char conv[])
{
  StableDistV *dist = (StableDistV *)args;
  double V[Nth];
  char check[Nx*Nth],checkmax[Nx*Nth];
  int k,l,t,P;

  if (stable_v_V_vec2(theta,Nth,args,V)==NULL) {
    printf("ERROR calculando V_vec2\n"); exit(2);
   }

//Bucles sin if para facil vectorizacion
  for (l=0;l<Nth;l++)
   {
    #ifdef DEBUG
    fprintf(FINTEG,"%e\t",theta);
    #endif
    for (k=0;k<Nx;k++)
      {
        t=k+Nx*l;
        intmax[k]=0;
        g[t] = dist->xxipow[k]*V[l];
        checkmax[t] = g[t] > 1.0;
        g[t] = g[t]*exp(-g[t]);
        check[t] = isnan(g[t]) || isinf(g[t]) || g[t]<0.0;
        #ifdef DEBUG
        fprintf(FINTEG,"%e\t",g[t]);
        #endif
      }
    #ifdef DEBUG
    fprintf(FINTEG,"\n");
    #endif
    }

//Tratamiento NaNs, infs, etc y situa el máximo en un intervalo entre las thetas dadas:
// alfa > 1 -> V decreciente -> si V*(x-xi)^a/(a-1) > 1 todavía no se ha alcanzado el máximo.
// alfa <=1 -> V creciente   -> si V*(x-xi)^a/(a-1) > 1 todavía no se ha alcanzado el máximo.
  P=dist->alfa>1.0;
  for (l=0;l<Nth;l++) {
    for (k=0;k<Nx;k++) {
      t=k+Nx*l;
      if(check[t]) g[t] = 0.0;
      if(P==checkmax[t]) intmax[k]++;
    }
  }

  return g;
}

double *
stable_v_pdf_g_mat2(const double *theta, void *args, size_t Nth, size_t Nx,
                  char conv[], double *g)
{
  StableDistV *dist = (StableDistV *)args;
  double *V;
  char *check;
  int k,l,t;

  V=(double*)malloc(sizeof(double)*Nth);
  check=(char*)malloc(sizeof(char)*Nth*Nx);

  if (stable_v_V_vec2(theta,Nth,args,V)==NULL) {
    printf("ERROR calculando V_vec2\n"); exit(2);
   }

//bucles sin if para facil vectorizacion
  for (l=0;l<Nth;l++)
   {
    #ifdef DEBUG
    fprintf(FINTEG,"%e\t",theta);
    #endif
    for (k=0;k<Nx;k++)
      {
        t=k+Nx*l;
        g[t] = dist->xxipow[k]*V[l];
        g[t] = g[t]*exp(-g[t]);
        check[t] = isnan(g[t]) || isinf(g[t]) || g[t]<0.0;
        #ifdef DEBUG
        fprintf(FINTEG,"%e\t",g[t]);
        #endif
      }
    #ifdef DEBUG
    fprintf(FINTEG,"\n");
    #endif
    }

//Tratamiento NaNs, infs, etc;
  for (l=0;l<Nth;l++) {
    for (k=0;k<Nx;k++) {
      t=k+Nx*l;
      if(check[t]) g[t] = 0.0;
    }
  }

  return g;
}

/*----------------------------------------------------------------------------*/
/*                             Parte Publica                                  */
/*----------------------------------------------------------------------------*/

double stable_v_get_relTOL() { return relTOL; }
void stable_v_set_relTOL(double value)
{
  relTOL = value;
  //FACTOR = pow(value,2.0/(SUBS-2.0));
  //FACTOR = 1e-16;
}
double stable_v_get_absTOL() { return absTOL; }
void stable_v_set_absTOL(double value)
{
  absTOL = value;
  //FACTOR = pow(value,2.0/(SUBS-2.0));
  //FACTOR = 1e-16;
}

double stable_v_get_ALFA_TH() { return ALFA_TH; }
void stable_v_set_ALFA_TH(double value) { ALFA_TH = value; }

double stable_v_get_BETA_TH() { return BETA_TH; }
void stable_v_set_BETA_TH(double value) { BETA_TH = value; }

double stable_v_get_XXI_TH() { return XXI_TH; }
void stable_v_set_XXI_TH(double value) { XXI_TH = value; }

double stable_v_get_THETA_TH() { return THETA_TH; }
void stable_v_set_THETA_TH(double value) { THETA_TH = value; }

FILE * stable_v_get_FINTEG() { return FINTEG; }
FILE * stable_v_set_FINTEG(char *name)
{
  FINTEG = fopen(name,"wt");
  return FINTEG;
}
FILE * stable_v_get_FLOG() { return FLOG; }
FILE * stable_v_set_FLOG(char * name)
{
  FLOG = fopen(name,"wt");
  return FLOG;
}
void stable_v_clear_LOG()
{
  char *name;

  name = "nombre";
  fclose(FLOG);
  FLOG = fopen(name,"wt");
}

int stable_v_setparams(StableDistV *dist,
                     double alfa, double beta, double sigma, double mu,
                     int parametrization)
{
  int zona;

  if(dist==NULL)
    {
      printf("ERROR");
      exit(2);
    }
  if((zona = stable_v_checkparams(alfa,beta,sigma,mu,parametrization)) == NOVALID)
    {
      printf ("Parametros no validos: %lf %lf %lf %lf %d\n",
             alfa, beta, sigma, mu, parametrization);
      return zona;
    }

  dist->alfa = alfa;
  dist->beta = beta;
  dist->sigma = sigma;

  switch (zona)
    {
      case STABLE:
        dist->alfainvalfa1 = alfa/(alfa-1.0);
        dist->xi = -beta*tan(0.5*alfa*M_PI);
        dist->theta0 = atan(-dist->xi)/alfa;
        //dist->k1 = -0.5/(alfa-1.0)*log(1.0+dist->xi*dist->xi);
        dist->k1 = pow(1.0+dist->xi*dist->xi,-0.5/(alfa-1.0));
        dist->S = pow(1.0+dist->xi*dist->xi,0.5/alfa);
        dist->Vbeta1 = pow(dist->alfa,-dist->alfainvalfa1) *
                       fabs(dist->alfa-1.0) * dist->k1;
        dist->stable_v_pdf_g = &stable_v_pdf_g2_;
        dist->stable_v_cdf_g = &stable_v_cdf_g2;
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
        break;

      case CAUCHY:
        dist->beta=0;
        dist->alfa=1;
        dist->c2_part = 0.0;
        dist->alfainvalfa1 = 0.0;
        dist->xi = 0.0;
        dist->theta0 = M_PI_2;
        //dist->k1 = log(2.0*M_1_PI);
        dist->k1 = 2.0*M_1_PI;
        dist->S = 2.0*M_1_PI;
        dist->c1 = 0.0;
        dist->c3 = M_1_PI;
        dist->Vbeta1=2.0*M_1_PI/M_E;
        dist->stable_v_pdf_g = &stable_v_pdf_g2_;/*cambio*/
//        dist->stable_v_cdf_g = &stable_v_cdf_g1;
        break;
      case ALFA_1:
        dist->alfa=1;
        dist->c2_part = 0.5/fabs(beta);
        dist->alfainvalfa1 = 0.0;
        dist->xi = 0.0;
        dist->theta0 = M_PI_2;
        //dist->k1 = log(2.0*M_1_PI);
        dist->k1 = 2.0*M_1_PI;
        dist->S = 2.0*M_1_PI;
        dist->c1 = 0.0;
        dist->c3 = M_1_PI;
        dist->Vbeta1=2.0*M_1_PI/M_E;
        dist->stable_v_pdf_g = &stable_v_pdf_g2_;/*cc*/
//        dist->stable_v_cdf_g = &stable_v_cdf_g1;
        break;
      case GAUSS:
        dist->alfa=2;
        dist->beta=0.0;
        dist->alfainvalfa1 = 2.0;
        dist->xi = 0.0;
        dist->theta0 = 0.0;
        //dist->k1 = log(2.0);
        dist->k1 = 2.0;
        dist->S = 2.0;
        dist->c1 = 1.0;
        dist->c2_part = 2.0*M_1_PI;
        dist->c3 = -M_1_PI;
        dist->stable_v_pdf_g = &stable_v_pdf_g2_;
        dist->stable_v_cdf_g = &stable_v_cdf_g2;
        break;
      case LEVY:
        dist->alfa = 0.5;
        dist->beta = (2.0*(beta>0)-1.0);//será 1 ó -1
        dist->alfainvalfa1 = -1.0;
        dist->xi = -dist->beta;
        dist->theta0 = 0.5*M_PI;
        //dist->k1 = 0.0;
        dist->k1 = 1.0;
        dist->S = 1.0;
        dist->c1 = 0.0;
        dist->c2_part = 0.5*M_1_PI;
        dist->c3 = M_1_PI;
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
  dist->xxipow = NULL;
  dist->ZONE = zona;

  return zona;
}

int stable_v_checkparams(double alfa, double beta, double sigma, double mu,
                       int parametrization)
{
  /*Comprobar parametros*/
  if (0.0 >= alfa || alfa > 2.0)
    {
      //printf("Alfa debe estar comprendido entre 0.0 y 2.0.");
      return NOVALID;
    }
  else if (beta < -1.0 || beta > 1.0)
    {
      //printf("Beta debe estar comprendido entre -1.0 y 1.0.");
      return NOVALID;
    }
  else if (sigma <= 0.0)
    {
      //printf("Sigma debe ser positivo.");
      return NOVALID;
    }
  else if (isnan(mu) || isnan(mu))
    {
      //printf("Mu debe ser un numero real.");
      return NOVALID;
    }
  else if (parametrization != 0 && parametrization != 1)
    {
      //printf("Solo se admiten parametrizaciones 0 y 1.");
      return NOVALID;
    }

  /*Determinacion de "zona"*/
  if ((2.0 - alfa) <= ALFA_TH)
    return GAUSS;
  else if (fabs(alfa-0.5) <= ALFA_TH && fabs((fabs(beta)-1.0)) <= BETA_TH)
    return LEVY;
  else if (fabs(alfa-1.0) <= ALFA_TH && fabs(beta) <= BETA_TH)
    return CAUCHY;
  else if (fabs(alfa-1.0) <= ALFA_TH)
    return ALFA_1;
  else
    return STABLE;
  /*Para alfa=1,beta=1 tenemos la ditribucion de Landau,
  pero se calcula con el mismo metodo*/
}

StableDistV * stable_v_create(double alfa, double beta, double sigma, double mu,
                           int parametrization)
{
  StableDistV *dist;
  //gsl_error_handler_t * old_handler;

  //old_handler = gsl_set_error_handler (&error_handler);

  dist = (StableDistV *) malloc(sizeof(StableDistV));
  if (dist == NULL)
    {
      perror("No se pudo crear la distribucion.");
      return NULL;
    }

  if((stable_v_setparams(dist,alfa,beta,sigma,mu,parametrization)) == NOVALID)
    {
      printf ("No se pudo crear la distribucion.");
      return NULL;
    }

  gsl_rng_env_setup(); //leemos las variables de entorno
  dist->gslworkspace = gsl_integration_workspace_alloc(IT_MAX);
  dist->gslrand = gsl_rng_alloc (gsl_rng_default);

  return dist;
}

StableDistV * stable_v_copy(StableDistV *src_dist)
{
  StableDistV *dist;

  dist = stable_v_create(src_dist->alfa, src_dist->beta,
                       src_dist->sigma, src_dist->mu_0, 0);
  return dist;
}

void stable_v_free(StableDistV *dist)
{
  if (dist == NULL)
    return;

  gsl_integration_workspace_free(dist->gslworkspace);
  gsl_rng_free(dist->gslrand);
  free(dist);
}
void reserva_mem(double **fde,double **q2,int N)
{
  *fde = (double*)malloc(N*2*sizeof(double));
  *q2 = (double*)malloc(N*sizeof(double));
}


void bucle_(StableDistV *dist,const double fa[],const double fc[],const double fb[], double fde[], double h,int N,
        double Q[],double q2[], double abserr[], double epsrel, double epsabs, int*warn, char conv[], char prevconv[])
{
  int k;

  for(k=0;k<N;k++)
    {
      if (prevconv[k]) {conv[k]=1; continue;}
      Q[k] = (h/6.0)*(fa[k]+4.0*fc[k]+fb[k]);
      q2[k] = (h/12.0)*(fa[k]+4.0*fde[k]+2.0*fc[k]+4.0*fde[k+N]+fb[k]);
      Q[k] = q2[k]+(q2[k]-Q[k])/15.0;
      //if(isnan(Q[k]) || isinf(Q[k])) { *warn = -1; exit(1);}
      abserr[k] = fabs(q2[k]-Q[k]);
      //if ((abserr[k]>epsabs) && (abserr[k] > Q[k]*epsrel)) {conv[k]=0;*warn=1;}
      conv[k] = abserr[k]<=epsabs || abserr[k]<=Q[k]*epsrel;
      *warn+=(!conv[k]);
    }
}

void
quadstep_vect(double*(*func)(double,void *,double*,size_t,unsigned int, unsigned int*),void *args,int N,
                double a, double b,const double fa[],const double fc[],const double fb[],
                double epsabs, const double epsrel, const unsigned int limit,
                int *warn, size_t *fcnt, double Q[], double abserr[], unsigned int first, unsigned int *next)//char prevconv[])
{
  double h,c,d,e,Q1;
  double *fde,*q2,*err;
//  char *conv;
  unsigned int *newnext;
  unsigned int newfirst,anterior;
  int k,warnac,warncb;

  fde = (double*)malloc(N*2*sizeof(double));
  q2 = (double*)malloc(N*sizeof(double));
//  conv = (char*)malloc(N*sizeof(char));
  newnext=(unsigned int *)malloc(N*sizeof(unsigned int));

  h = b-a;
  c = (a+b)*0.5;
  d = (a+c)*0.5;
  e = (c+b)*0.5;

//  func(d,args,fde,N,prevconv);
//  func(e,args,fde+N,N,prevconv);
  func(d,args,fde,N,first,next);
  func(e,args,fde+N,N,first,next);
  *fcnt+=2;

  *warn=0;
/* esto esta en la funcion bucle de arriba.
  for(k=0;k<N;k++)
    {
      if (prevconv[k]) {conv[k]=1; continue;}
      Q[k] = (h/6.0)*(fa[k]+4.0*fc[k]+fb[k]);
      q2[k] = (h/12.0)*(fa[k]+4.0*fde[k]+2.0*fc[k]+4.0*fde[k+N]+fb[k]);
      Q[k] = q2[k]+(q2[k]-Q[k])/15.0;
      //if(isnan(Q[k]) || isinf(Q[k])) { *warn = -1; exit(1);}
      abserr[k] = fabs(q2[k]-Q[k]);
      //if ((abserr[k]>epsabs) && (abserr[k] > Q[k]*epsrel)) {conv[k]=0;*warn=1;}
      conv[k] = abserr[k]<=epsabs || abserr[k]<=Q[k]*epsrel;
      *warn+=(!conv[k]);
    }
*/
//  bucle((StableDistV *)args,fa,fc,fb,fde,h,N,Q,q2,abserr,epsrel,epsabs,warn,conv,prevconv);

  anterior=first;
  newfirst=first;
  k=first;
  while (k<N)
  {
//      printf("%d\n",k);
      Q[k] = (h/6.0)*(fa[k]+4.0*fc[k]+fb[k]);
      Q1 = (h/12.0)*(fa[k]+4.0*fde[k]+2.0*fc[k]+4.0*fde[k+N]+fb[k]);
      Q[k] = Q1+(Q1-Q[k])/15.0;
      //if(isnan(Q[k]) || isinf(Q[k])) { *warn = -1; exit(1);}
      abserr[k] = fabs(Q1-Q[k]);
      //if ((abserr[k]>epsabs) && (abserr[k] > Q[k]*epsrel)) {conv[k]=0;*warn=1;}
      if(abserr[k]<=epsabs || abserr[k]<=Q[k]*epsrel)
      {
        if (k==newfirst) newfirst = next[k];
        else newnext[anterior] = next[k]; //si converge, hago que me salte el actual
      }
      else
      {
         newnext[anterior] = k; //si no converge, saltará a este k
         anterior = k; //actualizo 'anterior' al último sin converger.
         *warn=1;
      }
      k=next[k];
  }
  newnext[anterior]=N; //asi marcamos el nuevo final del newnext


  if (fabs(h)<2.22e-16) {*warn=1;}
  else if (*fcnt>limit) {*warn=2;}
  else if(*warn>0)
    {
      err = (double*)malloc(N*sizeof(double));
      quadstep_vect(func,args,N,a,c,fa,fde,fc,epsabs,epsrel,limit,&warnac,fcnt,Q,abserr,newfirst,newnext);
      quadstep_vect(func,args,N,c,b,fc,fde+N,fb,epsabs,epsrel,limit,&warncb,fcnt,q2,err,newfirst,newnext);
      *warn = (warnac>warncb) ? warnac : warncb;

//      for(k=0;k<N;k++)
      k=newfirst;
      while (k<N)
        {
          Q[k] = Q[k] + q2[k];
          abserr[k] = err[k]*err[k] + abserr[k]*abserr[k];
          abserr[k] = sqrt(abserr[k]);
          k=newnext[k];
        }

      free(err);
    }

  free(q2);
  free(fde);
  free(newnext);

}

void
stable_v_integration_v(StableDistV *dist,double *(function)(double,void*,double*,size_t,unsigned int,unsigned int *),
                   double a, double b, int N,
                   double epsabs, double epsrel, unsigned int limit,
                   double *result, double *abserr, int *warnt)
{
  double h,t[7];
  int k,warn[3];
  size_t fcnt=0;
  double *f,*Q,*err;
//char *conv;
  unsigned int *next;
  unsigned int first=0;

  f = (double*)malloc(N*7*sizeof(double));
//conv=(char*)malloc(N*sizeof(char));
  next=(unsigned int *)malloc(N*sizeof(unsigned int));

  for(k=0;k<N;k++) {next[k]=k+1;}

  t[3] = (a+b)*0.5;

  if (function(t[3],dist,f+N*3,N,first,next) == NULL) exit(1);

  h = 0.13579*(b - a);
  for(k=0;k<3;k++)
    {
      warn[k]=0;
      t[k] = a+k*h;
      if (function(t[k],dist,f+N*k,N,first,next) == NULL) exit(1);
      t[6-k] = b-k*h;
      if (function(t[6-k],dist,f+N*(6-k),N,first,next) == NULL) exit(1);
      //printf("  %+e\t%+e\n",t[k],t[6-k]);
    }
  fcnt=7;

  Q = (double*)malloc(N*3*sizeof(double));
  err = (double*)malloc(N*3*sizeof(double));
  quadstep_vect(function,(void*)dist,N,t[0],t[2],f    ,f+1*N,f+2*N,epsabs,epsrel,limit,warn,  &fcnt,Q,err,first,next);
  quadstep_vect(function,(void*)dist,N,t[2],t[4],f+2*N,f+3*N,f+4*N,epsabs,epsrel,limit,warn+1,&fcnt,Q+N,err+N,first,next);
  quadstep_vect(function,(void*)dist,N,t[4],t[6],f+4*N,f+5*N,f+6*N,epsabs,epsrel,limit,warn+2,&fcnt,Q+2*N,err+2*N,first,next);

  *warnt=0;
  for(k=0;k<N;k++)
    {
      result[k]=Q[k]+Q[k+N]+Q[k+N+N];
      abserr[k]=err[k]*err[k]+err[k+N]*err[k+N]+err[k+2*N]*err[k+2*N];
      abserr[k]=sqrt(abserr[k]);
      //printf("%d %e %e\n",k,result[k],abserr[k]);
    }

  for(k=0;k<3;k++)
    {
      if(warn[k]>*warnt) *warnt=warn[k];
    }

 // printf("%zu, %d, % 1.6e\n",fcnt,*warnt,result[1]);

  free(Q);
  free(err);
  free(f);
  free(next);
}

int comparator ( const void * elem1, const void * elem2 )
{
  double a=*(double*)elem1-*(double*)elem2;

  if (a<0) return -1;
  else if (a==0) return 0;
  else return 1;
}

void
stable_v_integration_v3(StableDistV *dist,double *(function)(double,void*,double*,size_t,unsigned int, unsigned int *),
                   double a, double b, int N,
                   double epsabs, double epsrel, unsigned int limit,
                   double *result, double *abserr, int *warnt)
{
  double *tauinit,*partialresult,*partialerr/*,*g*/;
  int k,l;

  int Nth_init=10;

  tauinit = (double*)malloc(Nth_init*sizeof(double));
  partialresult=(double*)malloc(N*sizeof(double));
  partialerr=(double*)malloc(N*sizeof(double));
  //g = (double*)malloc(N_thinit*N*sizeof(double));


  tauinit[0]=0;
  tauinit[Nth_init-1]=1;
  tauinit[(int)(0.5*(Nth_init-1))] = 0.250;

  for(k=0.5*(Nth_init-1)-1;k>0;k--)
    {
      tauinit[k] = tauinit[k+1]*0.5;
    }

  for(k=0.5*(Nth_init+1);k<Nth_init-1;k++)
    {
      tauinit[k] = 1.0-tauinit[(int)(Nth_init-1-k)];
    }

  //stable_v_pdf_g_mat2(tauinit,(void*)dist,15,N,NULL,g);

  for(k=0;k<N;k++)
    {
      result[k]=0.0;
      abserr[k]=0.0;
    }
  for(l=Nth_init-2;l>=0;l--)
//  for(l=0;l<Nth_init-1;l++)
    {
      stable_v_integration_v(dist,function,a+(b-a)*tauinit[l],a+(b-a)*tauinit[l+1], N,
                   epsabs/5, epsrel, limit, partialresult, partialerr,warnt);
      for(k=0;k<N;k++)
        {
          result[k]+=partialresult[k];
          abserr[k]+=partialerr[k]*partialerr[k];
        }
    }

  for(k=0;k<N;k++)
    {
      abserr[k]=sqrt(abserr[k])/result[k];
    }
}

void stable_v_cdf(StableDistV *dist, const double x[],
                const unsigned int Nx, double *pdf, double *err)
{
  int warn,k,k_l,k_r,k_xi;
  double theta[2];
  double *xxi,xxi_th;

  if (dist->ZONE != STABLE) { warn=stable_v_setparams(dist, 0.75, 0.5, dist->sigma, dist->mu_0, 0);}
  if (dist->alfa == 0.75 && dist->beta == 1.0) { printf("a");warn=stable_v_setparams(dist, 0.75, 0.5, dist->sigma, dist->mu_0, 0);}

  xxi = (double*)malloc(Nx*sizeof(double));
  dist->xxipow = (double*)malloc(Nx*sizeof(double));

  k_l=0;
  k_r=0;
  k_xi=0;
  //xxi_th = pow(10,-10/fabs(dist->alfainvalfa1));
  xxi_th = 1e-6;
  for(k=0;k<Nx;k++)
    {
      xxi[k]=(x[k]-dist->mu_0)/dist->sigma-dist->xi;
      if(fabs(xxi[k])<xxi_th)
        {
          k_xi++;
          dist->xxipow[k] = -1.0;
          pdf[k] = M_1_PI*(M_PI_2-dist->theta0);
          err[k] = 1e-16;
         // printf("punto en xi\n");
        }
      else if(xxi[k]<0)
        {
          k_l++;
          dist->xxipow[k] = pow(-xxi[k],dist->alfainvalfa1);
        }
      else
        {
          k_r++;
          dist->xxipow[k] = pow(xxi[k],dist->alfainvalfa1);
        }

      //printf("%e\n",dist->xxipow[k]);
    }

 // printf("%d %d %d\n",k_l,k_xi,k_r);

  if (k_l>0)
    {
      dist->theta0_= -dist->theta0;
      dist->beta_  = -dist->beta;

      theta[0] = -dist->theta0_+THETA_TH;
      theta[1] = M_PI_2-THETA_TH;

      stable_v_integration_v3(dist,dist->stable_v_cdf_g,theta[0],theta[1],k_l,
                           absTOL,relTOL,IT_MAX,pdf,err,&warn);
    }

  if (k_r>0)
    {
      dist->xxipow = dist->xxipow+k_l+k_xi; //se desplaza el puntero hacia la dcha
      dist->theta0_= dist->theta0;
      dist->beta_  = dist->beta;
      theta[0] = -dist->theta0_+THETA_TH;
      theta[1] = M_PI_2-THETA_TH;

      stable_v_integration_v3(dist,dist->stable_v_cdf_g,theta[0],theta[1],k_r,
                           absTOL,relTOL,IT_MAX,pdf+k_l+k_xi,err+k_l+k_xi,&warn);

      dist->xxipow = dist->xxipow-k_l-k_xi; //se devuelve a donde apuntaba
    }

  if (dist->alfa>1.0)
   {
    for (k=0;k<k_l;k++)
     {
      pdf[k]= -dist->c3*pdf[k];
     }
   }
  else
   {
    for (k=0;k<k_l;k++)
     {
      pdf[k]= 0.5 - (dist->theta0 + pdf[k])*M_1_PI;
     }
   }

  for (k=k_l+k_xi;k<Nx;k++)
    {
      pdf[k]=dist->c1+dist->c3*pdf[k];
    }

  free(dist->xxipow);
  free(xxi);
}

void stable_v_pdf(StableDistV *dist, const double x[],
                const unsigned int Nx, double *pdf, double *err)
{
  int warn,k,k_l,k_r,k_xi;
  double theta[2];
  double *xxi,xxi_th;

  if (dist->ZONE != STABLE) { warn=stable_v_setparams(dist, 0.75, 0.5, dist->sigma, dist->mu_0, 0);}

  xxi = (double*)malloc(Nx*sizeof(double));
  dist->xxipow = (double*)malloc(Nx*sizeof(double));

  k_l=0;
  k_r=0;
  k_xi=0;
  //xxi_th = pow(10,-10/fabs(dist->alfainvalfa1));
  xxi_th = 1e-6;
  for(k=0;k<Nx;k++)
    {
      xxi[k]=(x[k]-dist->mu_0)/dist->sigma-dist->xi;
      if(fabs(xxi[k])<xxi_th)
        {
          k_xi++;
          dist->xxipow[k] = -1.0;
          pdf[k] = exp(gammaln(1.0+1.0/dist->alfa)) * cos(dist->theta0)/(M_PI*dist->S)/dist->sigma;
          err[k] = 1e-16;
         // printf("punto en xi\n");
        }
      else if(xxi[k]<0)
        {
          k_l++;
          dist->xxipow[k] = pow(-xxi[k],dist->alfainvalfa1);
        }
      else
        {
          k_r++;
          dist->xxipow[k] = pow(xxi[k],dist->alfainvalfa1);
        }

      //printf("%e\n",dist->xxipow[k]);
    }

 // printf("%d %d %d\n",k_l,k_xi,k_r);

  if (k_l>0)
    {
      dist->theta0_= -dist->theta0;
      dist->beta_  = -dist->beta;

      theta[0] = -dist->theta0_+THETA_TH;
      theta[1] = M_PI_2-THETA_TH;

      stable_v_integration_v3(dist,dist->stable_v_pdf_g,theta[0],theta[1],k_l,
                           absTOL,relTOL,IT_MAX,pdf,err,&warn);
    }

  if (k_r>0)
    {
      dist->xxipow = dist->xxipow+k_l+k_xi;
      dist->theta0_= dist->theta0;
      dist->beta_  = dist->beta;
      theta[0] = -dist->theta0_+THETA_TH;
      theta[1] = M_PI_2-THETA_TH;

      stable_v_integration_v3(dist,dist->stable_v_pdf_g,theta[0],theta[1],k_r,
                           absTOL,relTOL,IT_MAX,pdf+k_l+k_xi,err+k_l+k_xi,&warn);

      dist->xxipow = dist->xxipow-k_l-k_xi;
    }

  for (k=0;k<k_l;k++)
    {
      pdf[k]=dist->c2_part/fabs(xxi[k])*pdf[k]/dist->sigma;
    }
  for (k=k_l+k_xi;k<Nx;k++)
    {
      pdf[k]=dist->c2_part/fabs(xxi[k])*pdf[k]/dist->sigma;
    }

  free(dist->xxipow);
  free(xxi);
}
unsigned int stable_v_get_THREADS() { return THREADS; }
void stable_v_set_THREADS(unsigned int value)
{
  if (value <= 0) THREADS = sysconf(_SC_NPROCESSORS_ONLN);
  else THREADS = value;
  //printf("\nCPUs = %u\n",THREADS);
}

unsigned int stable_v_get_SUBS() { return SUBS; }
void stable_v_set_SUBS(unsigned int value)
{
  SUBS = value;
  //stable_v_set_relTOL(relTOL); // Para que recalcule FACTOR
}

int stable_v_get_METHOD() { return METHOD; }
void stable_v_set_METHOD(int value) { METHOD = value; }
int stable_v_get_METHODNAME(char* name)
{
  switch (METHOD)
    {
      case STABLE_QAG2:
        return sprintf(name,
        "QAG2: Adaptative 21 point Gauss-Kronrod rule");
      case STABLE_QUADSTEP:
        return sprintf(name,
        "QUADSTEP: Adaptative Bisection");
      case STABLE_QROMBPOL:
        return sprintf(name,
        "QROMBPOL: Romberg with Polinomial Extrapolation");
      case STABLE_QROMBRAT:
        return sprintf(name,
        "ROMBRAT: Romberg with Rational Extrapolation");
    }

  sprintf(name,"Invalid method");
  return -1;
}
double
stable_v_rnd_value(StableDistV *dist)
{
  return dist->mu_1 +
         gsl_ran_levy_skew(dist->gslrand, dist->sigma, dist->alfa, dist->beta);
}

double *
stable_v_rnd(StableDistV *dist, const unsigned int n)
{
  double *rnd;
  int i;

  rnd = (double*)malloc(n*sizeof(double));
  if (rnd ==NULL) exit(2);

  for(i=0;i<n;i++)
    {
      rnd[i]=stable_v_rnd_value(dist);
    }
  return rnd;
}
