#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include "stable_api.h"
#include "benchmarking.h"

#define NUMTESTS 100

static void _measure_performance(StableDist *gpu_dist, double x, double alfa, double beta)
{
    int i;
    double gpu_pdf = 0;
    double gpu_start, gpu_end;
    double gpu_duration;
    double submits = 0, starts = 0, ends = 0, exec = 0;

    gpu_dist->cli.profile_enabled = 0;

    gpu_start = get_ms_time();
    for (i = 0; i < NUMTESTS; i++)
        gpu_pdf += stable_pdf_point(gpu_dist, x, NULL);
    gpu_end = get_ms_time();

    gpu_dist->cli.profile_enabled = 1;

    for (i = 0; i < NUMTESTS; i++)
    {
        gpu_pdf += stable_pdf_point(gpu_dist, x, NULL);
        submits += gpu_dist->cli.profiling.submit_acum;
        starts += gpu_dist->cli.profiling.start_acum;
        ends += gpu_dist->cli.profiling.finish_acum;
        exec += gpu_dist->cli.profiling.exec_time;
    }

    gpu_duration = (gpu_end - gpu_start) / NUMTESTS;
    submits /= NUMTESTS;
    starts /= NUMTESTS;
    ends /= NUMTESTS;
    exec /= NUMTESTS;
    gpu_pdf /= NUMTESTS * 2;

    printf("\r%.3f\t%.3f\t%.3f\t%.3g\t%.3g\t%.3g\t%.3g\n", alfa, beta, gpu_duration, submits, starts, ends, exec);
}

int main (void)
{
    double alfas[] = { 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75 };
    double betas[] = { 0.0, 0.25, 0.5, 0.75, 1.0, -0.25, -0.5, -0.75, -1.0 };
    double ev_points[] = { -1, 0, 1, -2, 2, 3, 4, 5, 6, 10, 100, 1000, -1000 };
    double sigma = 1.0, mu = 0.0;
    StableDist *gpu_dist;
    int ai, bi, evi;

    gpu_dist = stable_create(0.5, 0.0, 1, 0, 0);

    if (stable_activate_gpu(gpu_dist))
    {
        fprintf(stderr, "Couldn't activate GPU :(\n");
        return 1;
    }

    printf("=== GPU/CPU performance tests for libstable ===\n");

    stable_set_relTOL(1.2e-20);
    stable_set_absTOL(1e-10);

    for (ai = 0; ai < sizeof alfas / sizeof(double); ai++)
    {
        for (bi = 0; bi < sizeof betas / sizeof(double); bi++)
        {
            stable_setparams(gpu_dist, alfas[ai], betas[bi], sigma, mu, 0);

            for (evi = 0; evi < sizeof ev_points / sizeof(double); evi++)
                _measure_performance(gpu_dist, ev_points[evi], alfas[ai], betas[bi]);

        }
    }

    stable_free(gpu_dist);

    return 0;
}
