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


int main (int argc, char** argv)
{
    double alfa = 1.75, beta = 1, sigma = 1.0, mu = 0.0;
    int param = 0;
    double x[] = { 1 };
    double pdf[3] = { 0,0,0 }, gpu_pdf[3] = { 0,0,0 };
    double err[3] = { 0,0,0 }, gpu_err[3] = { 0,0,0 };
    size_t num_points = sizeof x / sizeof(double);
    int i;

    stable_clinteg_printinfo();

    if(argc >= 3)
    {
        alfa = strtod(argv[1], NULL);
        beta = strtod(argv[2], NULL);
    }

    if(argc >= 4)
    {
        x[0] = strtod(argv[3], NULL);
    }

    StableDist *dist = stable_create(alfa, beta, sigma, mu, param);

    if (!dist)
    {
        fprintf(stderr, "StableDist creation failure. Aborting.\n");
        return 1;
    }

    printf("Evaluating at α = %.3lf, β = %.3lf\n", alfa, beta);

    for(i = 0; i < num_points; i++)
        pdf[i] = stable_pdf_point(dist, x[i], err + i);


    if(stable_activate_gpu(dist))
    {
        fprintf(stderr, "Couldn't initialize GPU.\n");
        return 1;
    }

    if(stable_clinteg_points(&dist->cli, x, gpu_pdf, gpu_err, num_points, dist, clinteg_pdf))
    {
        fprintf(stderr, "Stable-OpenCL error. Aborting.\n");
        return 1;
    }

    for(i = 0; i < sizeof x / sizeof(double); i++)
    {
        double abserr = fabs(gpu_pdf[i] - pdf[i]);
        printf("PDF(%g;%1.2f,%1.2f,%1.2f,%1.2f) = %1.15e ± %1.2e\n",
           x[i], alfa, beta, sigma, mu, pdf[i], err[i]);
        printf("CPU relative error is %1.2e %%.\n\n", 100 * err[i] / pdf[i]);
        printf("GPU PDF(%g;%1.2f,%1.2f,%1.2f,%1.2f) = %1.15e ± %1.2e\n",
               x[i], alfa, beta, sigma, mu, gpu_pdf[i], gpu_err[i]);
        printf("GPU relative error is %1.2e %%.\n\n", 100 * fabs(gpu_err[i] / gpu_pdf[i]));
        printf("GPU / CPU difference: %3.3g abs, %3.3g rel\n\n", abserr, abserr / pdf[i]);
    }

    stable_free(dist);
    return 0;
}
