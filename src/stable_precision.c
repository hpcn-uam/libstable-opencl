/* benchmarks/stable_precision
 *
 * This benchmark program is employed to evaluate the precision of Libstable
 * when calculating the PDF or CDF of standard alpha-stable distributions.
 * It obtains several calculations of the desired function at:
 *     - A finite set of points in the alpha-beta parameter space.
 *     - A log-scaled swipe on abscissae axis.
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
#include "methods.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

int main(int argc, char *argv[])
{
	int n = 2, // 1 para pdf, 2 para cdf
		P = 0; // parametrizacion

	double gamma = 1,
		   delta = 0;

	int    THREADS = 16;
	double tol = 1.2e-14;
	double atol = 1e-50;

	int Na = 14,
		Nb = 5,
		Nx = 5417;

	double  alfa[] = {0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1, 1.01, 1.1, 1.25, 1.5, 1.75, 1.95, 2.0};
	double  beta[] = {0.0, 0.25, 0.5, 0.75, 1.0};
	double * x = (double*)malloc(Nx * sizeof(double));
	double x0[] = { -1000,  -100,      -10,          -1,        1,    10, 100};
	double Ndiv = 7, Nxdiv[] = {900,   720,      576,        1024,      576,   720, 901};
	unsigned int Nx0[]   = {  0,   900,     1620,        2196,     3220,  3796, 4516};
	double xD[] = {     1, 0.125, 0.015625, 0.001953125, 0.015625, 0.125,   1};

	double *pdf, *err;
	StableDist *dist = NULL;
	int i = 1, j, k;
	void(*func)(StableDist *, const double *, const int,
				double *, double *);
	struct timeval t_1, t_2/*,tp_1,tp_2*/;
	double t, tpdf;

	char name[256], nameerr[256]/*,nametiempos[256]*/;
	FILE * f;
	FILE * ferr;
	//  FILE * ftiempos;
	FILE * flog;
	FILE * finteg;

	if (argc != 3) {
		printf("Uso: stable_precision FUNCION RELTOL\n");
		exit(1);
	} else {
		tol = atof(argv[2]);
		n   = atoi(argv[1]);
	}

	for (i = 0; i < Ndiv; i++) {
		for (j = 0; j < Nxdiv[i]; j++)
			x[Nx0[i] + j] = x0[i] + xD[i] * j;
	}

	/*
	  for(i=0;i<Nx;i++)
	   {
	    printf("%1.9lf",x[i]);
	    if ((j+1)%10) printf("\t");
	    else printf("\n");
	   }

	  getchar();
	*/
	pdf = (double*)malloc(Nx * Nb * sizeof(double));
	err = (double*)malloc(Nx * Nb * sizeof(double));

	if (n == 1) {
		func = &stable_pdf;
		sprintf(name, "stable_precision_pdf_%1.1ea_%1.1er.txt", atol, tol);
		sprintf(nameerr, "stable_precision_pdf_err_%1.1ea_%1.1er.txt", atol, tol);
	} else if (n == 2) {
		func = &stable_cdf;
		sprintf(name, "stable_precision_cdf_%1.1ea_%1.1er.txt", atol, tol);
		sprintf(nameerr, "stable_precision_cdf_err_%1.1ea_%1.1er.txt", atol, tol);
	} else {
		printf("ERROR");
		exit(2);
	}

	if ((f = fopen(name, "wt")) == NULL) {
		printf("Error al crear archivo.");
		exit(2);
	}

	if ((ferr = fopen(nameerr, "wt")) == NULL) {
		printf("Error al crear archivo.");
		exit(2);
	}

	// Creacion de una distribucion estable
	if ((dist = stable_create(alfa[0], beta[0], gamma, delta, P)) == NULL) {
		printf("Error en la creacion de la distribucion");
		exit(EXIT_FAILURE);
	}

	finteg = stable_set_FINTEG("data_integrando.txt");
	flog = stable_set_FLOG("errlog.txt");

	stable_set_THREADS(THREADS);
	stable_set_relTOL(tol);
	stable_set_absTOL(atol);
	stable_set_METHOD(STABLE_QNG);
	stable_set_METHOD2(STABLE_QAG2);

	printf("FUNCTION=%d ALFAPOINTS=%d BETAPOINTS=%d XPOINTS=%d\n", n, Na, Nb, Nx);

	fprintf(f, "FUNCTION=%d ALFAPOINTS=%d BETAPOINTS=%d XPOINTS=%d\n", n, Na, Nb, Nx);
	fprintf(ferr, "FUNCTION=%d ALFAPOINTS=%d BETAPOINTS=%d XPOINTS=%d\n", n, Na, Nb, Nx);
	tpdf = 0;
	gettimeofday(&t_1, NULL);

	for (j = 0; j < Na; j++) {
		for (i = 0; i < Nb; i++) {
			stable_setparams(dist, alfa[j], beta[i], gamma, delta, P);
			//          gettimeofday(&tp_1,NULL);

			(func)(dist, (const double *)x, Nx, pdf + i * Nx, err + i * Nx);
			printf("        -> Alfa = %lf, Beta = %lf, %d points OK <-\n",
				   alfa[j], beta[i], Nx);
			//          gettimeofday(&tp_2,NULL);
			//          tpdf=tp_2.tv_sec-tp_1.tv_sec+(tp_2.tv_usec-tp_1.tv_usec)/1000000.0;
			//          fprintf(ftiempos,"%1.2lf %1.2lf %1.16lf %1.16lf\n",alfa[j],beta[i],tpdf,(double)Nx/tpdf);
			//          getchar();
		}

		//Volcado al archivo de texto
		fprintf(f, "\nAlfa = %lf\n________x\\beta________\t", alfa[j]);
		fprintf(ferr, "\nAlfa = %lf\n________x\\beta________\t", alfa[j]);

		for (i = 0; i < Nb; i++) {
			fprintf(f, "%1.16e\t", beta[i]);
			fprintf(ferr, "%1.16e\t", beta[i]);
		}

		fprintf(f, "\n");
		fprintf(ferr, "\n");

		for (k = 0; k < Nx; k++) {
			fprintf(f, "%1.16e\t", x[k]);
			fprintf(ferr, "%1.16e\t", x[k]);

			for (i = 0; i < Nb; i++) {
				fprintf(f, "%1.16e\t", *(pdf + i * Nx + k));
				fprintf(ferr, "%1.16e\t", *(err + i * Nx + k));
			}

			fprintf(f, "\n");
			fprintf(ferr, "\n");
		}

	}

	gettimeofday(&t_2, NULL);
	t = t_2.tv_sec - t_1.tv_sec + (t_2.tv_usec - t_1.tv_usec) / 1000000.0;

	printf("\n-----> Calculos completados en %3.3lf segundos <-----\n", t);

	fclose(f);
	fclose(ferr);
	//  fclose(ftiempos);
	fclose(finteg);
	fclose(flog);

	return 0;
}
