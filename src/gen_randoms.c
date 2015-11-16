/*
 * Copyright (C) 2015 - Naudit High Performance Computing and Networking
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
 */

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include "stable_api.h"
#include "benchmarking.h"
#include "opencl_integ.h"

int main (int argc, const char** argv)
{
    double alfa = 1.25;
    double beta = 0.7;
    size_t batches[] = { 100, 1000, 9000 };
    size_t num_batches = sizeof(batches) / sizeof(size_t);
    double* cpu_rands = NULL;
    double* gpu_rands = NULL;
    double start, end, tdiff;
    size_t i, batch_size;

    StableDist *dist = stable_create(alfa, beta, 1, 0, 0);

    if (!dist)
    {
        fprintf(stderr, "StableDist creation failure. Aborting.\n");
        return 1;
    }

    if(stable_activate_gpu(dist))
    {
        fprintf(stderr, "Couldn't initialize GPU.\n");
        return 1;
    }

    stable_set_absTOL(1e-20);
    stable_set_relTOL(1.2e-10);

    for (i = 0; i < num_batches; i++)
    {
        batch_size = batches[i];
        cpu_rands = calloc(batch_size, sizeof(double));
        gpu_rands = calloc(batch_size, sizeof(double));

        start = get_ms_time();
        stable_rnd(dist, cpu_rands, batch_size);
        end = get_ms_time();
        tdiff = end - start;

        fprintf(stderr, "%zu\t %.3lf %.3lf ", batch_size, tdiff, tdiff / batch_size);

        start = get_ms_time();
        stable_rnd_gpu(dist, gpu_rands, batch_size);
        end = get_ms_time();
        tdiff = end - start;

        fprintf(stderr, "%.3lf %.3lf\n", tdiff, tdiff / batch_size);
    }

    for(i = 0; i < 0; i++)
        printf("%.6lf\n", gpu_rands[i]);

    stable_free(dist);
    return 0;
}
