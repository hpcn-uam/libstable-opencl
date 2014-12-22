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
    double x = 10;
    double pdf = 0, gpu_pdf = 0, dummy = 0;
    double dummy_expect = 135492.0634920634920634920634920634920635;
    double err;
    int i;
    int max_tries = 1;

    printf("=== GPU tests for libstable:\n");
    printf("Using %d points GK rule with %d subdivisions.\n", GK_POINTS, GK_SUBDIVISIONS);
    printf("Precision used: %s.\n\n", cl_precision_type);

    StableDist *dist = stable_create(alfa, beta, sigma, mu, param);

    if (!dist)
    {
        fprintf(stderr, "StableDist creation failure. Aborting.\n");
        return 1;
    }

    for(i = 0; i < max_tries; i++)
        pdf += stable_pdf_point(dist, x, &err);

    pdf /= max_tries;

    printf("PDF(%g;%1.2f,%1.2f,%1.2f,%1.2f) = %1.15e ± %1.2e\n",
           x, alfa, beta, sigma, mu, pdf, err);
    printf("CPU relative error is %1.2e %%.\n\n", 100 * err / pdf);

    if(stable_activate_gpu(dist))
    {
        fprintf(stderr, "Couldn't initialize GPU.\n");
        return 1;
    }

    for(i = 0; i < max_tries; i++)
        gpu_pdf += stable_pdf_point(dist, x, &err);

    gpu_pdf /= max_tries;

    printf("GPU PDF(%g;%1.2f,%1.2f,%1.2f,%1.2f) = %1.15e ± %1.2e\n",
           x, alfa, beta, sigma, mu, gpu_pdf, err);
    printf("GPU relative error is %1.2e %%.\n\n", 100 * fabs(err / gpu_pdf));

    printf("GPU / CPU difference: %3.3g\n\n", fabs(gpu_pdf - pdf));

    printf("Testing now dummy integrand...\n");
    dist->ZONE = GPU_TEST_INTEGRAND_SIMPLE;
    stable_clinteg_integrate(&dist->cli, 0, 2, 0, 0, 0, &dummy, &err, dist);

    printf("Difference between GPU dummy result (%lf ± %1.2e) and expected result (%d): %lf\n\n",
         dummy, err, 2, 2 - dummy);


    printf("Testing now test integrand...\n");
    dist->ZONE = GPU_TEST_INTEGRAND;
    stable_clinteg_integrate(&dist->cli, -5, 5, 0, 0, 0, &dummy, &err, dist);

    printf("Difference between GPU test result (%lf ± %1.2e) and expected result (%lf): %lf\n",
         dummy, err, dummy_expect, dummy_expect - dummy);

    stable_free(dist);
    return 0;
}
