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


int main (void)
{
    double bc_start, bc_end;
    double alfa = 1.25, beta = 0.5, sigma = 1.0, mu = 0.0;
    int param = 0;
    double x = 10;
    double pdf = 0, gpu_pdf = 0;
    int i;
    int max_tries = 1;

    StableDist *dist = stable_create(alfa, beta, sigma, mu, param);

    if (!dist)
    {
        fprintf(stderr, "StableDist creation failure. Aborting.\n");
        return 1;
    }

    BENCHMARK_BEGIN;
    for(i = 0; i < max_tries; i++)
        pdf += stable_pdf_point(dist, x, NULL);
    BENCHMARK_END(max_tries, "PDF CPU");

    pdf /= max_tries;

    printf("PDF(%g;%1.2f,%1.2f,%1.2f,%1.2f) = %1.15e\n",
           x, alfa, beta, sigma, mu, pdf);
    

    stable_activate_gpu(dist);

    BENCHMARK_BEGIN;
    for(i = 0; i < max_tries; i++)
        gpu_pdf += stable_pdf_point(dist, x, NULL);
    BENCHMARK_END(max_tries, "PDF GPU");

    gpu_pdf /= max_tries;

    printf("GPU PDF(%g;%1.2f,%1.2f,%1.2f,%1.2f) = %1.15e\n",
           x, alfa, beta, sigma, mu, gpu_pdf);

    printf("GPU / CPU difference: %3.3g\n", fabs(gpu_pdf - pdf));

    stable_free(dist);
    return 0;
}
