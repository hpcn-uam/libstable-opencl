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

static void _measure_gpu_performance(StableDist *gpu_dist, double x, double alfa, double beta, struct opencl_profile* general_profile)
{
    int i;
    double gpu_pdf = 0;
    double gpu_start, gpu_end, gpu_duration;
    struct opencl_profile current_prof_info;

    bzero(&current_prof_info, sizeof(struct opencl_profile));
    gpu_dist->cli.profile_enabled = 0;

    gpu_start = get_ms_time();
    for (i = 0; i < NUMTESTS; i++)
        gpu_pdf += stable_pdf_point(gpu_dist, x, NULL);
    gpu_end = get_ms_time();

    gpu_duration = gpu_end - gpu_start;

    gpu_dist->cli.profile_enabled = 1;

    gpu_start = get_ms_time();
    for (i = 0; i < NUMTESTS; i++)
    {
        gpu_pdf += stable_pdf_point(gpu_dist, x, NULL);
        profile_sum(&current_prof_info, &gpu_dist->cli.profiling);
    }
    gpu_end = get_ms_time();

    current_prof_info.profile_total = gpu_end - gpu_start;
    current_prof_info.total = gpu_duration;

    profile_sum(general_profile, &current_prof_info);

    current_prof_info.total /= NUMTESTS;
    current_prof_info.submit_acum /= NUMTESTS;
    current_prof_info.start_acum /= NUMTESTS;
    current_prof_info.finish_acum /= NUMTESTS;
    current_prof_info.exec_time /= NUMTESTS;
    current_prof_info.argset /= NUMTESTS;
    current_prof_info.enqueue /= NUMTESTS;
    current_prof_info.buffer_read /= NUMTESTS;
    current_prof_info.set_results /= NUMTESTS;

    fprintf(stderr, "%.3f %.3f\t| %.3f | %.5f %.5f %.5f %.5f | %.5f %.5f %.5f %.5f\n", alfa, beta, current_prof_info.total,
          current_prof_info.submit_acum, current_prof_info.start_acum, current_prof_info.finish_acum, current_prof_info.exec_time,
           current_prof_info.argset, current_prof_info.enqueue, current_prof_info.buffer_read, current_prof_info.set_results);
}

static void _measure_cpu_performance(StableDist *dist, double x, double alfa, double beta, struct opencl_profile* general_profile)
{
    int i;
    double gpu_pdf = 0;
    double gpu_start, gpu_end;

    dist->cli.profile_enabled = 0;

    gpu_start = get_ms_time();
    for (i = 0; i < NUMTESTS; i++)
        gpu_pdf += stable_pdf_point(dist, x, NULL);
    gpu_end = get_ms_time();

    general_profile->total += gpu_end - gpu_start;

    printf("\r%.3f %.3f\t| %.3f |\n", alfa, beta, (gpu_end - gpu_start) / NUMTESTS);
 }

int main (int argc, char** argv)
{
    double alfas[] = { 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75 };
    double betas[] = { 0.0, 0.25, 0.5, 0.75, 1.0, -0.25, -0.5, -0.75, -1.0 };
    double ev_points[] = { -1, 0, 1, -2, 2, 3, 4, 5, 6, 10, 100, 1000, -1000 };
    double sigma = 1.0, mu = 0.0;
    size_t alfas_len = sizeof alfas / sizeof(double);
    size_t betas_len = sizeof betas / sizeof(double);
    size_t evpoints_len = sizeof ev_points / sizeof(double);
    long test_num;
    StableDist *dist;
    int ai, bi, evi;
    double cpu_total;
    struct opencl_profile profile;
    double ms_per_point;

    if(argc > 1)
    {
        alfas_len = 1;
        betas_len = 1;
    }

    test_num = alfas_len * betas_len * evpoints_len * NUMTESTS;

    dist = stable_create(0.5, 0.0, 1, 0, 0);
    bzero(&profile, sizeof profile);
    printf("=== GPU/CPU performance tests for libstable ===\n");

    stable_set_relTOL(1.2e-20);
    stable_set_absTOL(1e-10);

    fprintf(stderr, "α     β\t\t| time  |\n");

    for (ai = 0; ai < alfas_len; ai++)
    {
        for (bi = 0; bi < betas_len; bi++)
        {
            stable_setparams(dist, alfas[ai], betas[bi], sigma, mu, 0);

            for (evi = 0; evi < evpoints_len; evi++)
                _measure_cpu_performance(dist, ev_points[evi], alfas[ai], betas[bi], &profile);
        }
    }

    if (stable_activate_gpu(dist))
    {
        fprintf(stderr, "Couldn't activate GPU :(\n");
        return 1;
    }

    cpu_total = profile.total;
    bzero(&profile, sizeof profile);

    fprintf(stderr, "α     β\t\t| time  | submit  start   finish  total   | argset  enqueue bufread setres\n");

    for (ai = 0; ai < alfas_len; ai++)
    {
        for (bi = 0; bi < betas_len; bi++)
        {
            stable_setparams(dist, alfas[ai], betas[bi], sigma, mu, 0);

            for (evi = 0; evi < evpoints_len; evi++)
                _measure_gpu_performance(dist, ev_points[evi], alfas[ai], betas[bi], &profile);

        }
    }

    fprintf(stderr, "α     β\t\t| time  | submit  start   finish  total   | argset  enqueue bufread setres\n");
    ms_per_point = profile.total / test_num;
    printf("\nTest finished: %ld total points.\n", test_num);
    printf("GPU: %10.2f pps, %.5f ms per point.\n", 1000 * test_num / profile.total, ms_per_point);
    printf("CPU: %10.2f pps, %.5f ms per point.\n", 1000 * test_num / cpu_total, cpu_total / test_num);
    printf("\nDetailed GPU data:\n");
    printf("Parameter\t| ms per point\t| %% point time\n");
    printf("Kernel time\t| %.5f\t| %.2f\n", profile.exec_time / test_num, 100 * profile.exec_time / profile.profile_total);
    printf("Buffer reading\t| %.5f\t| %.2f\n", profile.buffer_read / test_num, 100 * profile.buffer_read / profile.profile_total);
    printf("Argset\t\t| %.5f\t| %.2f\n", profile.argset / test_num, 100 * profile.argset / profile.profile_total);
    printf("H->D transfer\t| %.5f\t| %.2f\n", (profile.start_acum) / test_num, 100 * (profile.start_acum) / profile.profile_total);
    stable_free(dist);

    return 0;
}
