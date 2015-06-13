#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include "stable_api.h"
#include "benchmarking.h"
#include "opencl_integ.h"

int main (void)
{
    double alfas[] = { 0.25, 0.5, 0.75, 1.25 };
    double betas[] = { 0, 0.5, 1 };
    double intervals[] = { -50, 50 };
    int points_per_interval = 100;
    double cpu_pdf[points_per_interval], gpu_pdf[points_per_interval];
    double cpu_err[points_per_interval], gpu_err[points_per_interval];

    printf("=== GPU tests for libstable:\n");
    printf("Using %d points GK rule with %d subdivisions.\n", GK_POINTS, GK_SUBDIVISIONS);
    printf("Precision used: %s.\n\n", cl_precision_type);

    StableDist *dist = stable_create(0.5, 0, 1, 0, 0);

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

    size_t ai, bi, i, j;
    double points[points_per_interval];
    size_t interval_count = (sizeof(intervals) / sizeof(double)) - 1;
    size_t alfa_count = sizeof(alfas) / sizeof(double);
    size_t beta_count = sizeof(betas) / sizeof(double);

    double abs_diff_sum, rel_diff_sum, gpu_err_sum;

    for(i = 0; i < interval_count; i++)
    {
        double begin = intervals[i];
        double end = intervals[i + 1];
        double step = (end - begin) / points_per_interval;

        printf("\n=== Interval (%.0lf, %.0lf)\n", begin, end);

        for(j = 0; j < points_per_interval; j++)
            points[j] = j * step + begin;

        for(ai = 0; ai < alfa_count; ai++)
        {
            for(bi = 0; bi < beta_count; bi++)
            {
                stable_setparams(dist, alfas[ai], betas[bi], 1, 0, 0);

                stable_pdf_gpu(dist, points, points_per_interval, gpu_pdf, gpu_err);
                stable_pdf(dist, points, points_per_interval, cpu_pdf, cpu_err);

                abs_diff_sum = 0;
                rel_diff_sum = 0;
                gpu_err_sum = 0;

                for(j = 0; j < points_per_interval; j++)
                {
                    double cpu = cpu_pdf[j];
                    double gpu = gpu_pdf[j];
                    double diff = fabs(cpu - gpu);

                    gpu_err_sum += fabs(gpu_err[j]);
                    abs_diff_sum += diff;

                    if(cpu != 0)
                        rel_diff_sum += diff / cpu;
                }

                abs_diff_sum /= points_per_interval;
                rel_diff_sum /= points_per_interval;
                gpu_err_sum /= points_per_interval;

                printf("%.3lf %.3lf %1.3g   \t%1.3g\t%1.3g\n", alfas[ai], betas[bi], abs_diff_sum, rel_diff_sum, gpu_err_sum);
            }
        }
    }

    stable_free(dist);
    return 0;
}
