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

typedef void (*evaluator)(StableDist *, const double*, const int, double *, double *);

static void _measure_gpu_performance(StableDist *gpu_dist, double* x, size_t nx, double alfa, double beta, struct opencl_profile* general_profile, evaluator fn)
{
    int i;
    double gpu_start, gpu_end, gpu_duration;
    struct opencl_profile current_prof_info;
    double* dummya, *dummyb;

    dummya = calloc(nx, sizeof(double));
    dummyb = calloc(nx, sizeof(double));

    bzero(&current_prof_info, sizeof(struct opencl_profile));
    gpu_dist->cli.profile_enabled = 0;

    gpu_start = get_ms_time();
    for (i = 0; i < NUMTESTS; i++)
        stable_clinteg_points(&gpu_dist->cli, x, dummya, dummyb, nx, gpu_dist, clinteg_pdf);
    gpu_end = get_ms_time();

    gpu_duration = gpu_end - gpu_start;

    gpu_dist->cli.profile_enabled = 1;

    gpu_start = get_ms_time();
    for (i = 0; i < NUMTESTS; i++)
    {
        fn(gpu_dist, x, nx, dummya, dummyb);
        profile_sum(&current_prof_info, &gpu_dist->cli.profiling);
    }
    gpu_end = get_ms_time();

    current_prof_info.profile_total = gpu_end - gpu_start;
    current_prof_info.total = gpu_duration;

    profile_sum(general_profile, &current_prof_info);

    current_prof_info.total /= NUMTESTS * nx;
    current_prof_info.submit_acum /= NUMTESTS * nx;
    current_prof_info.start_acum /= NUMTESTS * nx;
    current_prof_info.finish_acum /= NUMTESTS * nx;
    current_prof_info.exec_time /= NUMTESTS * nx;
    current_prof_info.argset /= NUMTESTS * nx;
    current_prof_info.enqueue /= NUMTESTS * nx;
    current_prof_info.buffer_read /= NUMTESTS * nx;
    current_prof_info.set_results /= NUMTESTS * nx;

    fprintf(stdout, "%.3f %.3f\t| %.3f | %.5f %.5f %.5f %.5f | %.5f %.5f %.5f %.5f\n", alfa, beta, current_prof_info.total,
            current_prof_info.submit_acum, current_prof_info.start_acum, current_prof_info.finish_acum, current_prof_info.exec_time,
            current_prof_info.argset, current_prof_info.enqueue, current_prof_info.buffer_read, current_prof_info.set_results);
}

static void _measure_cpu_performance(StableDist *dist, double* x, size_t nx, double alfa, double beta, struct opencl_profile* general_profile, evaluator fn)
{
    int i;
    double gpu_start, gpu_end;
    double* pdf, *err;

    pdf = calloc(nx, sizeof(double));
    err = calloc(nx, sizeof(double));

    dist->cli.profile_enabled = 0;

    gpu_start = get_ms_time();
    for (i = 0; i < NUMTESTS; i++)
        fn(dist, x, nx, pdf, err);
    gpu_end = get_ms_time();

    general_profile->total += gpu_end - gpu_start;

    printf("%.3f %.3f\t| %.3f |\n", alfa, beta, (gpu_end - gpu_start) / (NUMTESTS * nx));
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
    clinteg_type type = clinteg_pdf;
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
        else if (strcmp(argv[i], "cdf") == 0)
        {
            type = clinteg_cdf;
        }
    }

    if (type == clinteg_pdf)
    {
        cpu_fn = stable_pdf;
        gpu_fn = stable_pdf_gpu;
        printf(" PDF precision testing\n");
    }
    else
    {
        cpu_fn = stable_cdf;
        gpu_fn = stable_cdf_gpu;
        printf(" CDF precision testing\n");
    }

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

    for (ai = 0; ai < alfas_len; ai++)
    {
        for (bi = 0; bi < betas_len; bi++)
        {
            stable_setparams(dist, alfas[ai], betas[bi], sigma, mu, 0);
            _measure_cpu_performance(dist, ev_points, evpoints_len, alfas[ai], betas[bi], &profile, cpu_fn);
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

    fprintf(stdout, "α     β\t\t| time  | submit  start   finish  total   | argset  enqueue bufread setres\n");

    for (ai = 0; ai < alfas_len; ai++)
    {
        for (bi = 0; bi < betas_len; bi++)
        {
            stable_setparams(dist, alfas[ai], betas[bi], sigma, mu, 0);

            _measure_gpu_performance(dist, ev_points, evpoints_len, alfas[ai], betas[bi], &profile, gpu_fn);
        }
    }


    fflush(stdout);

    fprintf(stdout, "α     β\t\t| time  | submit  start   finish  total   | argset  enqueue bufread setres\n");
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
