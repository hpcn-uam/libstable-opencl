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

#define NUMTESTS 100

void profile_sum(struct opencl_profile* dest, struct opencl_profile* src)
{
    dest->submit_acum += src->submit_acum;
    dest->start_acum += src->start_acum;
    dest->finish_acum += src->finish_acum;
    dest->exec_time += src->exec_time;
    dest->argset += src->argset;
    dest->enqueue += src->enqueue;
    dest->buffer_read += src->buffer_read;
    dest->set_results += src->set_results;
    dest->total += src->total;
    dest->profile_total += src->profile_total;
}

typedef void (*evaluator)(StableDist *, const double, double *);

static void _measure_performance(StableDist *dist, double* x, size_t nx, double alfa, double beta, struct opencl_profile* general_profile, evaluator fn)
{
    int i;
    double gpu_start, gpu_end;
    double* pdf, *err;

    pdf = calloc(nx, sizeof(double));
    err = calloc(nx, sizeof(double));

    dist->cli.profile_enabled = 0;

    gpu_start = get_ms_time();
    for (i = 0; i < nx; i++)
        fn(dist, x[i], err);
    gpu_end = get_ms_time();

    general_profile->total += gpu_end - gpu_start;

    printf("%.3f %.3f\t| %.3f |\n", alfa, beta, (gpu_end - gpu_start) / nx);
}

static void fill(double* array, double begin, double end, size_t size)
{
    size_t i;
    double step = (end - begin) / size;

    for (i = 0; i < size; i++)
        array[i] = begin + i * step;
}

int main (int argc, char** argv)
{
    size_t alfas_len = 20;
    size_t betas_len = 10;
    size_t evpoints_len = 500;
    double alfas[alfas_len];
    double betas[betas_len];
    double ev_points[evpoints_len];
    double sigma = 1.0, mu = 0.0;
    long test_num;
    StableDist *dist;
    int ai, bi;
    double cpu_total;
    struct opencl_profile profile;
    double ms_per_point;
    short enable_cpu = 0;
    evaluator cpu_fn;
    evaluator gpu_fn;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "fast") == 0)
        {
            alfas_len = 10;
            betas_len = 5;
            evpoints_len = 80;
        }
        else if (strcmp(argv[i], "flash") == 0)
        {
            alfas_len = 1;
            betas_len = 1;
        }
        else if(strcmp(argv[i], "cpu") == 0)
        {
            enable_cpu = 1;
        }
    }


    cpu_fn = (evaluator) stable_inv_point;
    gpu_fn = (evaluator) stable_inv_point_gpu;

    fill(alfas, 0, 2, alfas_len);
    fill(betas, 0, 1, betas_len);
    fill(ev_points, -100, 100, evpoints_len);

    test_num = alfas_len * betas_len * evpoints_len * NUMTESTS;

    dist = stable_create(0.5, 0.0, 1, 0, 0);
    bzero(&profile, sizeof profile);

    stable_clinteg_printinfo();

    stable_set_relTOL(1.2e-20);
    stable_set_absTOL(1e-10);

    fprintf(stdout, "α     β\t\t| time  |\n");

    if(enable_cpu)
    {
        for (ai = 0; ai < alfas_len; ai++)
        {
            for (bi = 0; bi < betas_len; bi++)
            {
                stable_setparams(dist, alfas[ai], betas[bi], sigma, mu, 0);
                _measure_performance(dist, ev_points, evpoints_len, alfas[ai], betas[bi], &profile, cpu_fn);
            }
        }
    }

    fflush(stdout);

    if (stable_activate_gpu(dist))
    {
        fprintf(stderr, "Couldn't activate GPU :(\n");
        return 1;
    }

    cpu_total = profile.total;
    bzero(&profile, sizeof profile);

    fprintf(stdout, "α     β\t\t| time  |\n");

    for (ai = 0; ai < alfas_len; ai++)
    {
        for (bi = 0; bi < betas_len; bi++)
        {
            stable_setparams(dist, alfas[ai], betas[bi], sigma, mu, 0);
            _measure_performance(dist, ev_points, evpoints_len, alfas[ai], betas[bi], &profile, gpu_fn);
        }
    }


    fflush(stdout);

    fprintf(stdout, "α     β\t\t| time  |\n");
    ms_per_point = profile.total / test_num;
    printf("\nTest finished: %ld total points.\n", test_num);
    printf("GPU: %10.2f pps, %.5f ms per point.\n", 1000 * test_num / profile.total, ms_per_point);
    printf("CPU: %10.2f pps, %.5f ms per point.\n", 1000 * test_num / cpu_total, cpu_total / test_num);

    stable_free(dist);

    return 0;
}
