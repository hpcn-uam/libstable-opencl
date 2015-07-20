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
    double cdf[3] = { 0,0,0 }, gpu_cdf[3] = { 0,0,0 };
    double pdf_err[3] = { 0,0,0 }, gpu_pdf_err[3] = { 0,0,0 };
    double cdf_err[3] = { 0,0,0 }, gpu_cdf_err[3] = { 0,0,0 };
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

    stable_pdf(dist, x, 3, pdf, pdf_err);
    stable_cdf(dist, x, 3, cdf, cdf_err);

    if(stable_activate_gpu(dist))
    {
        fprintf(stderr, "Couldn't initialize GPU.\n");
        return 1;
    }

    if(stable_clinteg_points(&dist->cli, x, gpu_pdf, NULL, gpu_pdf_err, num_points, dist, clinteg_pdf))
    {
        fprintf(stderr, "Stable-OpenCL error. Aborting.\n");
        return 1;
    }

    if(stable_clinteg_points(&dist->cli, x, NULL, gpu_cdf, gpu_cdf_err, num_points, dist, clinteg_cdf))
    {
        fprintf(stderr, "Stable-OpenCL error. Aborting.\n");
        return 1;
    }

    for(i = 0; i < sizeof x / sizeof(double); i++)
    {
        double abspdf_err = fabs(gpu_pdf[i] - pdf[i]);
        printf("PDF(%g;%1.2f,%1.2f,%1.2f,%1.2f) = %1.15e, relerr %1.2e\n",
           x[i], alfa, beta, sigma, mu, pdf[i], pdf_err[i]);
        printf("GPU PDF(%g;%1.2f,%1.2f,%1.2f,%1.2f) = %1.15e, relerr %1.2e\n",
               x[i], alfa, beta, sigma, mu, gpu_pdf[i], gpu_pdf_err[i]);
        printf("PDF GPU / CPU difference: %3.3g abs, %3.3g rel\n\n", abspdf_err, abspdf_err / pdf[i]);

        double abscdf_err = fabs(gpu_cdf[i] - cdf[i]);

        printf("CDF(%g;%1.2f,%1.2f,%1.2f,%1.2f) = %1.15e, relerr %1.2e\n",
           x[i], alfa, beta, sigma, mu, cdf[i], cdf_err[i]);
        printf("GPU CDF(%g;%1.2f,%1.2f,%1.2f,%1.2f) = %1.15e, relerr %1.2e\n",
               x[i], alfa, beta, sigma, mu, gpu_cdf[i], gpu_cdf_err[i]);
        printf("CDF GPU / CPU difference: %3.3g abs, %3.3g rel\n\n", abscdf_err, abscdf_err / cdf[i]);
    }

    stable_free(dist);
    return 0;
}
