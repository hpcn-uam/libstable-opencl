/*
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

static int _dbl_compare (const void * a, const void * b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;

    return (db < da) - (da < db);
}

int main (int argc, const char** argv)
{
    double alfas[] = { 0.25, 0.5, 0.75, 1.25, 1.5 };
    double betas[] = { 0, 0.5, 1 };
    double intervals[] = { -1000, -100, 100, 1000 };
    int points_per_interval = 1000;
    double cpu_vals[points_per_interval], gpu_vals[points_per_interval];
    double cpu_err[points_per_interval], gpu_err[points_per_interval];
    clinteg_mode mode = mode_pdf;

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

    if(argc > 1 && strcmp("cdf", argv[1]) == 0)
        mode = mode_cdf;

    if(mode == mode_pdf)
        printf(" PDF precision testing\n");
    else
        printf(" CDF precision testing\n");

    stable_set_absTOL(1e-20);
    stable_set_relTOL(1.2e-10);

    size_t ai, bi, i, j;
    double points[points_per_interval];
    double rel_errs[points_per_interval];
    double abs_errs[points_per_interval];
    size_t interval_count = (sizeof(intervals) / sizeof(double)) - 1;
    size_t alfa_count = sizeof(alfas) / sizeof(double);
    size_t beta_count = sizeof(betas) / sizeof(double);
    size_t in_cpu_bounds_count;
    double percentage_in_bounds;
    ssize_t total_in_cpu_bounds_count = 0;
    double total_relerr = 0, total_abserr = 0;
    ssize_t total_points;
    FILE* f = fopen("precision.dat", "w");

    double abs_diff_sum, rel_diff_sum, gpu_err_sum, cpu_err_sum;

    for(i = 0; i < interval_count; i++)
    {
        double begin = intervals[i];
        double end = intervals[i + 1];
        double step = (end - begin) / points_per_interval;

        printf("\n=== Interval (%.0lf, %.0lf)\n", begin, end);
        printf("alfa  beta   abserr    relerr    Mabserr   Mrelerr   gpuerr    cpuerr    within bounds\n");

        for(j = 0; j < points_per_interval; j++)
            points[j] = j * step + begin;

        for(ai = 0; ai < alfa_count; ai++)
        {
            for(bi = 0; bi < beta_count; bi++)
            {
                stable_setparams(dist, alfas[ai], betas[bi], 1, 0, 0);

                if(mode == mode_pdf)
                {
                    stable_pdf_gpu(dist, points, points_per_interval, gpu_vals, gpu_err);
                    stable_pdf(dist, points, points_per_interval, cpu_vals, cpu_err);
                }
                else
                {
                    stable_cdf_gpu(dist, points, points_per_interval, gpu_vals, gpu_err);
                    stable_cdf(dist, points, points_per_interval, cpu_vals, cpu_err);
                }

                abs_diff_sum = 0;
                rel_diff_sum = 0;
                gpu_err_sum = 0;
                cpu_err_sum = 0;
                in_cpu_bounds_count = 0;

                for(j = 0; j < points_per_interval; j++)
                {
                    double cpu = cpu_vals[j];
                    double gpu = gpu_vals[j];
                    double diff = fabs(cpu - gpu);
                    double rel_diff = 0;

                    gpu_err_sum += fabs(gpu_err[j]);

                    if(!isnan(cpu_err[j]))
                        cpu_err_sum += fabs(cpu_err[j]);

                    if(!isnan(cpu))
                        abs_diff_sum += diff;

                    if(cpu != 0 && !isnan(cpu) && cpu > 1e-20)
                        rel_diff = diff / cpu;

                    rel_errs[j] = rel_diff;
                    abs_errs[j] = diff;

                    rel_diff_sum += rel_diff;
                    fprintf(f, "%.2lf %.2lf %5.3g %9.5g %9.5g %9.5g %9.5g %9.5g\n", alfas[ai], betas[bi], points[j], cpu_vals[j], gpu_vals[j], diff, rel_diff, fabs(gpu_err[j]));

                    if(diff < cpu_err[j] || diff == 0)
                        in_cpu_bounds_count++;
                }

                total_relerr += rel_diff_sum;
                total_abserr += abs_diff_sum;

                abs_diff_sum /= points_per_interval;
                rel_diff_sum /= points_per_interval;
                gpu_err_sum /= points_per_interval;
                cpu_err_sum /= points_per_interval;

                total_in_cpu_bounds_count += in_cpu_bounds_count;
                percentage_in_bounds = 100 * ((double)in_cpu_bounds_count) / points_per_interval;

                qsort(abs_errs, points_per_interval, sizeof(double), _dbl_compare);
                qsort(rel_errs, points_per_interval, sizeof(double), _dbl_compare);

                printf("%.3lf %.3lf  %8.3g  %8.3g  %8.3g  %8.3g  %8.3g  %8.3g  %8.1lf %%\n",
                    alfas[ai], betas[bi],
                    abs_diff_sum, rel_diff_sum,
                    abs_errs[points_per_interval / 2], rel_errs[points_per_interval / 2],
                    gpu_err_sum, cpu_err_sum,
                    percentage_in_bounds);
            }
        }
    }

    total_points = points_per_interval * interval_count * alfa_count * beta_count;
    percentage_in_bounds = 100 * ((double)total_in_cpu_bounds_count / total_points);
    total_relerr /= total_points;
    total_abserr /= total_points;

    printf("\nTotal percentage of points within bounds: %.3lf %%\n", percentage_in_bounds);
    printf("Average relerr: %g, abserr: %g\n", total_relerr, total_abserr);
    stable_free(dist);
    return 0;
}
