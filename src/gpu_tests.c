/* tests/introexample
 *
 * Simple example program to introduce the use of Libstable.
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
#include <sys/time.h>
#include <stdlib.h>
#include "stable_api.h"
#include "benchmarking.h"
#include "opencl_integ.h"


int main (void)
{
    double alfa = 1.25, beta = 0.5, sigma = 1.0, mu = 0.0;
    int param = 0;
    double x[] = { 2, 1, 0 };
    double pdf[3] = { 0,0,0 }, gpu_pdf[3] = { 0,0,0 };
    double err[3] = { 0,0,0 }, gpu_err[3] = { 0,0,0 };
    int i;

    printf("=== GPU tests for libstable:\n");
    printf("Using %d points GK rule with %d subdivisions.\n", GK_POINTS, GK_SUBDIVISIONS);
    printf("Precision used: %s.\n\n", cl_precision_type);

    StableDist *dist = stable_create(alfa, beta, sigma, mu, param);

    if (!dist)
    {
        fprintf(stderr, "StableDist creation failure. Aborting.\n");
        return 1;
    }

    for(i = 0; i < sizeof x / sizeof(double); i++)
        pdf[i] = stable_pdf_point(dist, x[i], err + i);


    if(stable_activate_gpu(dist))
    {
        fprintf(stderr, "Couldn't initialize GPU.\n");
        return 1;
    }

    stable_clinteg_points(&dist->cli, x, gpu_pdf, gpu_err, 3, dist);
    for(i = 0; i < sizeof x / sizeof(double); i++)
    {
        printf("PDF(%g;%1.2f,%1.2f,%1.2f,%1.2f) = %1.15e ± %1.2e\n",
           x[i], alfa, beta, sigma, mu, pdf[i], err[i]);
        printf("CPU relative error is %1.2e %%.\n\n", 100 * err[i] / pdf[i]);
        printf("GPU PDF(%g;%1.2f,%1.2f,%1.2f,%1.2f) = %1.15e ± %1.2e\n",
               x[i], alfa, beta, sigma, mu, gpu_pdf[i], gpu_err[i]);
        printf("GPU relative error is %1.2e %%.\n\n", 100 * fabs(gpu_err[i] / gpu_pdf[i]));
        printf("GPU / CPU difference: %3.3g\n\n", fabs(gpu_pdf[i] - pdf[i]));
    }

    stable_free(dist);
    return 0;
}
