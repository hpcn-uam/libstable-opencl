/* tests/print_arch
 *
 * Auxiliary function to plot outputs from other example program on the
 * text console.
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
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int print_arch(char name[], int D, int F, int C)
{

	double CperV, maxx = -1e300, min = 1e300;
	double *valor = NULL, *x = NULL;
	int kmin, kmax, prev_y, N, j, k, nread;
	//int n;
	double *y;
	FILE *file;
	char c_o, c_m;
	char buf[128];

	c_o = '.';
	c_m = '#';

	if (C <= 0) C = 80;

	if (F <= 0) F = 20;

	file = fopen(name, "rt");

	N = 0;

	while (!feof(file)) {
		valor = (double *)realloc((double *)valor, (N + 1) * sizeof(double));
		x = (double *)realloc((double *)x, (N + 1) * sizeof(double));
		nread = fscanf(file, "%lf\t", &x[N]);
		nread = fscanf(file, "%lf\t", &valor[N]);

		if (fgets(buf, sizeof(buf), file) == NULL) break;

		N++;
	}

	CperV = N / (double)C;

	if (CperV < 1) {
		CperV = 1;
		C = N;
	}

	y = (double*)calloc(C, sizeof(double));

	for (j = 0; j < C; j++) {
		kmin = (int)(j * CperV);

		if (kmin > N - 1) kmin = N - 1;

		kmax = (int)((j + 1) * CperV);

		if (kmax > N) kmax = N;

		for (k = kmin; k < kmax; k++) {
			y[j] += valor[k];
			//printf("Valor[%i]=%f\t,y[%i]=%f\n",j,valor[j],j,y[j]);
		}

		y[j] = y[j] / (kmax - kmin);

		if (y[j] < min) min = y[j];

		if (y[j] > maxx) maxx = y[j];

		//printf("kmin=%i, kmax=%i, y[%i]=%f\n",kmin,kmax,j,y[j]);
	}

	for (k = 0; k < C; k++) {
		y[k] = F * (1.0 - (y[k] - min) / (maxx - min));
		y[k] -= fmod(y[k], 1.0);

		if (y[k] == F) y[k]--;

		//printf("y[%i]=%f\n",k,y[k]);
	}

	//	printf("-----------------------------\n");
	//	printf("N=%i C=%i min=%f max=%f CperV=%f\n",N,C,min,maxx,CperV);
	//	printf("-----------------------------\n");

	prev_y = 0;

	for (j = 0; j < F; j++) {
		printf("% 1.2e ", (double)(F - 1 - j) / (F - 1) * (maxx - min) + min);

		for (k = 0; k < C; k++) {
			if (y[k] == j && prev_y == 0) {
				printf("%c", c_m);
				prev_y = 1;
			} else if (y[k] != j && prev_y == 0) {
				printf("%c", c_o);
				prev_y = 1;
			} else if (prev_y == 1 && y[k] == j) {
				printf("\b*");
				prev_y = 1;
			}

			prev_y = 0;
		}

		printf("\n");
	}

	if (x[0] > 0.0) printf(" ");

	printf("    %1.3lf", x[0]);

	for (j = 0; j < C - 11; j++) printf(" ");

	if (x[N - 1] > 0.0) printf(" ");

	printf("%1.3lf\n", x[N - 1]);
	free(valor);
	free(y);
	fclose(file);
	return 0;
}
