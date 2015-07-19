#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include "stable_api.h"
#include "benchmarking.h"
#include "opencl_integ.h"

int main (int argc, const char** argv)
{
    double alfas[] = { 0.25, 0.5, 0.75, 1.25, 1.5 };
    double betas[] = { 0, 0.5, 1 };
    double intervals[] = { -1000, -100, 100, 1000 };
    int points_per_interval = 1000;
    double cdf_vals[points_per_interval];

    stable_clinteg_printinfo();

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

    stable_set_absTOL(1e-20);
    stable_set_relTOL(1.2e-10);

    size_t ai, bi, i, j;
    double points[points_per_interval];
    size_t interval_count = (sizeof(intervals) / sizeof(double)) - 1;
    size_t alfa_count = sizeof(alfas) / sizeof(double);
    size_t beta_count = sizeof(betas) / sizeof(double);
    size_t in_cpu_bounds_count;
    double total_relerr = 0, total_abserr = 0;

    double abs_diff_sum, rel_diff_sum, gpu_err_sum, cpu_err_sum;

    for(i = 0; i < interval_count; i++)
    {
        double begin = intervals[i];
        double end = intervals[i + 1];
        double step = (end - begin) / points_per_interval;

        printf("\n=== Interval (%.0lf, %.0lf)\n", begin, end);
        printf("alfa  beta   abserr\n");

        for(j = 0; j < points_per_interval; j++)
            points[j] = j * step + begin;

        for(ai = 0; ai < alfa_count; ai++)
        {
            for(bi = 0; bi < beta_count; bi++)
            {
                stable_setparams(dist, alfas[ai], betas[bi], 1, 0, 0);
                stable_cdf_gpu(dist, points, points_per_interval, cdf_vals, NULL);

                abs_diff_sum = 0;
                rel_diff_sum = 0;
                gpu_err_sum = 0;
                cpu_err_sum = 0;
                in_cpu_bounds_count = 0;

                for(j = 0; j < points_per_interval; j++)
                {
                    double guess = stable_inv_point_gpu(dist, cdf_vals[j], NULL);
                    double diff = fabs(guess - points[j]);

                    abs_diff_sum += diff;
                }

                total_relerr += rel_diff_sum;
                total_abserr += abs_diff_sum;

                abs_diff_sum /= points_per_interval;

                printf("%.3lf %.3lf  %8.3g\n",
                    alfas[ai], betas[bi],
                    abs_diff_sum);
            }
        }
    }


    stable_free(dist);
    return 0;
}
