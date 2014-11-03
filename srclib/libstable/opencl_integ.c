#include "stable_api.h"
#include "opencl_common.h"
#include "benchmarking.h"

#include <limits.h>
#include <stdio.h>
#include <stdarg.h>

#ifndef max
#define max(a,b) (a < b ? b : a)
#endif


static int _stable_set_results(struct stable_clinteg *cli);

static int _stable_can_overflow(struct stable_clinteg *cli)
{
    cl_uint work_threads = cli->points_rule * cli->subdivisions;

    return work_threads / cli->subdivisions != cli->points_rule;
}

int stable_clinteg_init(struct stable_clinteg *cli)
{
    int err;

    cli->points_rule = GK_POINTS;
    cli->subdivisions = GK_SUBDIVISIONS;

#ifdef BENCHMARK
    cli->profile_enabled = 1;
#endif

    if (_stable_can_overflow(cli))
    {
        stablecl_log(log_warning, "[Stable-OpenCl] Warning: possible overflow in work dimension (%d x %d).\n"
                     , cli->points_rule, cli->subdivisions);
        return -1;
    }

    if (opencl_initenv(&cli->env, "opencl/stable_pdf.cl", "stable_pdf"))
    {
        stablecl_log(log_message, "[Stable-OpenCl] OpenCL environment failure.\n");
        return -1;
    }

    cli->subinterval_errors = (cl_precision *) calloc(cli->subdivisions, sizeof(cl_precision));

    if (!cli->subdivisions)
    {
        perror("[Stable-OpenCl] Host memory allocation failed.");
        return -1;
    }

    cli->gauss = clCreateBuffer(cli->env.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                sizeof(cl_precision) * cli->subdivisions, NULL, &err);
    cli->kronrod = clCreateBuffer(cli->env.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                  sizeof(cl_precision) * cli->subdivisions, NULL, &err);
    cli->args = clCreateBuffer(cli->env.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                               sizeof(struct stable_info), NULL, &err);

    if (err)
    {
        stablecl_log(log_err, "[Stable-OpenCl] Buffer creation failed: %s\n", opencl_strerr(err));
        return -1;
    }

    cli->h_gauss = clEnqueueMapBuffer(cli->env.queue, cli->gauss, CL_FALSE, CL_MAP_READ, 0, cli->subdivisions * sizeof(cl_precision), 0, NULL, NULL, &err);
    cli->h_kronrod = clEnqueueMapBuffer(cli->env.queue, cli->kronrod, CL_FALSE, CL_MAP_READ, 0, cli->subdivisions * sizeof(cl_precision), 0, NULL, NULL, &err);
    cli->h_args = clEnqueueMapBuffer(cli->env.queue, cli->args, CL_TRUE, CL_MAP_WRITE, 0, sizeof(struct stable_info), 0, NULL, NULL, &err);

    if (err)
    {
        stablecl_log(log_err, "[Stable-OpenCl] Buffer mapping failed: %s. "
                     "Host pointers (gauss, kronrod, args): (%p, %p, %p)\n",
                     opencl_strerr(err), cli->h_gauss, cli->h_kronrod, cli->h_args);
        return -1;
    }

    return 0;
}

void stable_retrieve_profileinfo(struct stable_clinteg *cli, cl_event event)
{
    stablecl_profileinfo(&cli->profiling, event);
}

static void _stable_print_profileinfo(struct opencl_profile *prof)
{
    printf("OpenCL Profile: %3.3g ms submit, %3.3g ms start, %3.3g ms finish.\n", prof->submit_acum, prof->start_acum, prof->finish_acum);
    printf("\tKernel exec time: %3.3g.\n", prof->exec_time);
}

double stable_clinteg_integrate(struct stable_clinteg *cli, double a, double b, double epsabs, double epsrel, unsigned short limit,
                                double *result, double *abserr, struct StableDistStruct *dist)
{
    cl_int err = 0;
    size_t work_threads, workgroup_size;
    cl_event event;

    // TODO: Create a "StableParams" structure that holds all of this parameters
    //      and use that instead of embedding all of them in the general StableDist
    //      structure, so I can avoid all this useless copying.
    cli->h_args->beta_ = dist->beta_;
    cli->h_args->k1 = dist->k1;
    cli->h_args->xxipow = dist->xxipow;
    cli->h_args->ibegin = a;
    cli->h_args->iend = b;
    cli->h_args->theta0_ = dist->theta0_;
    cli->h_args->alfa = dist->alfa;
    cli->h_args->alfainvalfa1 = dist->alfainvalfa1;

    cli->h_args->subinterval_length = (b - a) / (double) cli->subdivisions;
    cli->h_args->half_subint_length = cli->h_args->subinterval_length / 2;
    cli->h_args->threads_per_interval = cli->points_rule / 2 + 1 + 1; // Extra thread for sum.
    cli->h_args->gauss_points = (cli->points_rule / 2 + 1) / 2;
    cli->h_args->kronrod_points = cli->points_rule / 2;

    if (dist->ZONE == ALFA_1)
        cli->h_args->integrand = PDF_ALPHA_EQ1;
    else
        cli->h_args->integrand = PDF_ALPHA_NEQ1;

    stablecl_log(log_message, "[Stable-OpenCL] Integration begin - interval (%.3lf, %.3lf), %d subdivisions, %e subinterval length, %u threads per interval.\n",
                 a, b, cli->subdivisions, cli->h_args->subinterval_length, cli->h_args->threads_per_interval);

    err = clEnqueueWriteBuffer(cli->env.queue, cli->args, CL_FALSE, 0, sizeof(struct stable_info), cli->h_args, 0, NULL, NULL);

    bench_begin(cli->profiling.argset, cli->profile_enabled);
    int argc = 0;
    err |= clSetKernelArg(cli->env.kernel, argc++, sizeof(cl_mem), &cli->gauss);
    err |= clSetKernelArg(cli->env.kernel, argc++, sizeof(cl_mem), &cli->kronrod);
    err |= clSetKernelArg(cli->env.kernel, argc++, sizeof(cl_mem), &cli->args);
    bench_end(cli->profiling.argset, cli->profile_enabled);

    if (err)
    {
        stablecl_log(log_err, "[Stable-OpenCl] Couldn't set kernel arguments: error %d\n", err);
        goto cleanup;
    }
   

    workgroup_size = 64; // Minimum accepted number, it seems.
    work_threads = max(cli->h_args->threads_per_interval, workgroup_size) * cli->subdivisions; // We already checked for overflow.

    stablecl_log(log_message, "[Stable-OpenCl] Enqueing kernel - %zu work threads, %zu workgroup size (%d points per interval)\n", work_threads, workgroup_size, cli->points_rule);

    bench_begin(cli->profiling.enqueue, cli->profile_enabled);
    err = clEnqueueNDRangeKernel(cli->env.queue, cli->env.kernel,
                                 1, NULL, &work_threads, &workgroup_size, 0, NULL, &event);
    bench_end(cli->profiling.enqueue, cli->profile_enabled);

    if (err)
    {
        stablecl_log(log_err, "[Stable-OpenCl] Error enqueueing the kernel command: %s (%d)\n", opencl_strerr(err), err);
        goto cleanup;
    }

    bench_begin(cli->profiling.buffer_read, cli->profile_enabled);
    err |= clEnqueueReadBuffer(cli->env.queue, cli->gauss, CL_FALSE, 0, sizeof(cl_precision) * cli->subdivisions,
                               cli->h_gauss, 0, NULL, NULL);
    err |= clEnqueueReadBuffer(cli->env.queue, cli->kronrod, CL_TRUE, 0, sizeof(cl_precision) * cli->subdivisions,
                               cli->h_kronrod, 0, NULL, NULL);
    bench_end(cli->profiling.buffer_read, cli->profile_enabled);

    if (err)
    {
        stablecl_log(log_err, "[Stable-OpenCl] Error reading results from the GPU: %d\n", err);
        goto cleanup;
    }
   
    if (cli->profile_enabled)
        stable_retrieve_profileinfo(cli, event);


    bench_begin(cli->profiling.set_results, cli->profile_enabled);
    _stable_set_results(cli);
    bench_end(cli->profiling.set_results, cli->profile_enabled);

    *result = cli->result;
    *abserr = cli->abs_error;

cleanup:
    stablecl_log(log_message, "[Stable-OpenCl] Integration end.\n");

    return err;
}

static int _stable_set_results(struct stable_clinteg *cli)
{
    double gauss_sum = 0, kronrod_sum = 0;
    unsigned int i;

    for (i = 0; i < cli->subdivisions; i++)
    {
        stablecl_log(log_message, "[Stable-OpenCl] Interval %d: G %.3e K %.3e\n", i, cli->h_gauss[i], cli->h_kronrod[i]);
        gauss_sum += (double) cli->h_gauss[i];
        kronrod_sum += (double) cli->h_kronrod[i];
        cli->subinterval_errors[i] = cli->h_gauss[i] - cli->h_kronrod[i];
    }

    cli->result = kronrod_sum;
    cli->abs_error = kronrod_sum - gauss_sum;

    stablecl_log(log_message, "[Stable-OpenCl] Results set: gauss_sum = %.3g, kronrod_sum = %.3g, abserr = %.3g\n", gauss_sum, kronrod_sum, cli->abs_error);
    return 0;
}

void stable_clinteg_teardown(struct stable_clinteg *cli)
{
    clEnqueueUnmapMemObject(cli->env.queue, cli->gauss, cli->h_gauss, 0, NULL, NULL);
    clEnqueueUnmapMemObject(cli->env.queue, cli->kronrod, cli->h_kronrod, 0, NULL, NULL);
    clEnqueueUnmapMemObject(cli->env.queue, cli->args, cli->h_args, 0, NULL, NULL);

    clReleaseMemObject(cli->gauss);
    clReleaseMemObject(cli->kronrod);
    clReleaseMemObject(cli->args);

    opencl_teardown(&cli->env);
}


