/* benchmarks/stable_performance
 * 
 * This benchmark program is employed to evaluate the performance of Libstable
 * when calculating the PDF or CDF of standard alpha-stable distributions.
 * It obtains several calculations of the desired function at:
 *     - A finite set of points in the alpha-beta parameter space.
 *     - A log-scaled swipe on abscissae axis.
 *     - Different precisions
 *     - Different number of threads of execution
 *
 * Each computation is repeated several times to obtain confidence intervals on
 * Libstable performance.
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "stable_api.h"

#include <sys/time.h>
#include <time.h>

int main(int argc, char *argv[])
{
  long int n,n0,n1,ka,kb,kx,kt,kp,npuntosacum=0,i,j,
               P = 0,
               Na=7,
               Nb = 3,
               Ndiv=1,
               Nx=10001,//10002,
               Ntest=100,//20,
               Nxdiv[]={10001},//{5001,5001},
               Nx0[]  ={0},//{  0, 5001},
               Np=5,
               threads=16;
  double alfa[]={0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75},
         beta[]={0.0, 0.5, 1.0},
         xN[]  ={/*-1002,*/-2,998.06},
         xD[]  ={/* 0.2,*/0.1},
         gamma = 1.0,
         delta = 0.0,
         reltol[] = {1.2e-14, 1e-12, 1e-10, 1e-8, 1e-6},
         abstol = 1e-50,
         t_total=0, tpdf;
  double *pdf,*err, * x;
 
  FILE * ftiempos, *finteg, *flog;

  struct timeval t_ini,t_fin,tp_1,tp_2;

  char marker[]={'-','\\','|','/'};
  char nametiempos[256];

  time_t rawtime;
  struct tm * timeinfo;
  char hora[20];

  StableDist *dist;
  void (*func)(StableDist *, const double *, const int,double *, double *);

  if (argc!=3)
   {
    printf("Uso: stable_performance FUNC THREADS\n     FUNC=1: PDF\n     FUNC=2: CDF\n     FUNC=0: Ambas\n");
    exit(1);
   }
  else
   {
    n=atoi(argv[1]);
    threads = atoi(argv[2]);
   }

  pdf = (double*)malloc(Nx*sizeof(double));
  err = (double*)malloc(Nx*sizeof(double));
  x   = (double*)malloc(Nx*sizeof(double));

  for (i=0;i<Ndiv;i++)
   {
    for (j=0;j<Nxdiv[i];j++)
     {
      x[Nx0[i]+j]=xN[i]+xD[i]*j;
     }
   }

  // Creacion de una distribucion estable	
  if((dist = stable_create(alfa[0],beta[0],gamma,delta,P)) == NULL)
    {
      printf("Error en la creacion de la distribucion");
      exit(EXIT_FAILURE);
    }

  finteg = stable_set_FINTEG("data_integrando.txt");
  flog = stable_set_FLOG("errlog.txt");

  stable_set_THREADS(threads);
  stable_set_relTOL(reltol[0]);
  stable_set_absTOL(abstol);
  stable_set_METHOD(STABLE_QNG);
  stable_set_METHOD2(STABLE_QAG2);

  printf("\n\n          FUNCION     RELTOL     ALFA   BETA   INTERVALO\n");

  if (n==0) {
    n0=1;
    n1=2;
   }
  else if (n==1 || n==2) {
    n0=n;
    n1=n;
   }
  else {
    printf ("n debe ser 0, 1 o 2\n");
    exit(3);
   }

for(n=n0;n<=n1;n++)
 {
  if (n==2)
   {
    func = &stable_cdf;
    sprintf(nametiempos,"stable_performance_cdf_%ldhilo_%1.1ea.txt",threads,abstol);
   }
  else
   {
    func = &stable_pdf;
    sprintf(nametiempos,"stable_performance_pdf_%ldhilo_%1.1ea.txt",threads,abstol);
   }

  if((ftiempos=fopen(nametiempos, "wt"))==NULL)
    {
      printf("Error al crear archivo.");
      exit(2);
    }

  npuntosacum=0;
  gettimeofday(&t_ini,NULL);
  fprintf(ftiempos,"Np %ld Nalfa %ld Nbeta %ld Nintervalos %ld Ntest %ld\n",Np,Na,Nb,Ndiv,Ntest);
  fprintf(ftiempos,"Nxporintervalo"); for(kx=0;kx<Ndiv;kx++) {fprintf(ftiempos," %ld",Nxdiv[kx]);} fprintf(ftiempos,"\n");
  fprintf(ftiempos,"FUNCION %ld\n",n);
  for(kp=0;kp<Np;kp++)
   {
    stable_set_relTOL(reltol[kp]);
    fprintf(ftiempos," RELTOL %1.2e\n",reltol[kp]);
    
    for(ka=0;ka<Na;ka++)
     {
      fprintf(ftiempos,"  ALFA %1.2lf\n",alfa[ka]);
      for(kb=0;kb<Nb;kb++)
       {
        stable_setparams(dist,alfa[ka],beta[kb],gamma,delta,P);
        fprintf(ftiempos,"   BETA %1.2lf\n",beta[kb]);
	time(&rawtime);
        timeinfo = localtime(&rawtime);
	    strftime (hora,20,"%X",timeinfo);
        printf(" %s      %ld   % 1.2e   % 1.2lf  % 1.2lf   ",
                       hora,n,reltol[kp],alfa[ka],beta[kb]);

        for(kx=0;kx<Ndiv;kx++)
         {
          fprintf(ftiempos,"%s INTERVALO %ld ",hora,kx);
          for(kt=0;kt<Ntest;kt++)
           {
            putchar(marker[kt%4]);
			fflush(stdout);
            
            gettimeofday(&tp_1,NULL);
            (func)(dist,(const double *)x,Nxdiv[kx],pdf,err);
            gettimeofday(&tp_2,NULL);
            tpdf=tp_2.tv_sec-tp_1.tv_sec+(tp_2.tv_usec-tp_1.tv_usec)/1000000.0;
      
            fprintf(ftiempos,"% 1.16lf ",tpdf);

            npuntosacum+=Nxdiv[kx];

            printf("\b");
           }
		   fprintf(ftiempos,"\n");
           printf("%ld ",kx);
         }
        printf("\n");
       }
     }
   }

  gettimeofday(&t_fin,NULL);
  t_total=t_fin.tv_sec-t_ini.tv_sec+(t_fin.tv_usec-t_ini.tv_usec)/1000000.0;
  fprintf(ftiempos,"\n%ld puntos calculados en %3.3lf segundos\n",npuntosacum,t_total);
  fprintf(ftiempos,"              Datos en nolan_performance.txt\n");
  fprintf(ftiempos,"  Rendimiento: %ld puntos / %3.3lf s = %lf puntos/s\n",npuntosacum,t_total,((double)npuntosacum)/t_total); 
  fclose(ftiempos);
  ftiempos=NULL;

  printf("\n-----> %ld puntos calculados completados en %3.3lf segundos <-----\n",npuntosacum,t_total);
  printf("              Datos en nolan_performance.txt\n");
  printf(" Rendimiento: %ld puntos / %3.3lf segundos = %lf puntos/segundo\n",npuntosacum,t_total,((double)npuntosacum)/((double)t_total));
 }

  ftiempos=NULL;

  return 0;
}

