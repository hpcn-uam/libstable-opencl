#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include "stable_api.h"
#include "benchmarking.h"

#define NUMTESTS 40

static void _measure_performance(StableDist *cpu_dist, StableDist *gpu_dist, double x)
{
    int i;
    double cpu_pdf = 0, gpu_pdf = 0;
    double cpu_start, cpu_end, gpu_start, gpu_end;
    double gpu_duration, cpu_duration;

    cpu_start = get_ms_time();
    for (i = 0; i < NUMTESTS; i++)
        cpu_pdf += stable_pdf_point(cpu_dist, x, NULL);
    cpu_end = get_ms_time();

    gpu_start = get_ms_time();
    for (i = 0; i < NUMTESTS; i++)
        gpu_pdf += stable_pdf_point(gpu_dist, x, NULL);
    gpu_end = get_ms_time();

    gpu_duration = (gpu_end - gpu_start) / NUMTESTS;
    cpu_duration = (cpu_end - cpu_start) / NUMTESTS;

    printf("%3.5f\t%3.5f\n", cpu_duration, gpu_duration);
}

int main (void)
{
    double alfas[] = { 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75 };
    double betas[] = { 0.0, 0.5, 1.0 };
    double ev_points[] = { 1, 0, -1, 10, 1000, -1000 };
    double sigma = 1.0, mu = 0.0;
    StableDist *cpu_dist, *gpu_dist;
    int ai, bi, evi;

    printf("=== GPU/CPU performance tests for libstable ===\n");

    for (ai = 0; ai < sizeof alfas / sizeof(double); ai++)
    {
        for (bi = 0; bi < sizeof betas / sizeof(double); bi++)
        {   
            printf("[α: %.2f\t β: %.2f\t]\n", alfas[ai], betas[bi]);

            cpu_dist = stable_create(alfas[ai], betas[bi], sigma, mu, 0);
            gpu_dist = stable_create(alfas[ai], betas[bi], sigma, mu, 0);
            stable_activate_gpu(gpu_dist);

            for (evi = 0; evi < sizeof ev_points / sizeof(double); evi++)
                _measure_performance(cpu_dist, gpu_dist, ev_points[evi]);

            stable_free(cpu_dist);
            stable_free(gpu_dist);
        }
    }

    return 0;
}
