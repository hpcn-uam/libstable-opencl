#include "stable_api.h"
#include "opencl_common.h"

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
    cli->points_rule = GK_POINTS;
    cli->subdivisions = GK_SUBDIVISIONS;

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

    cli->h_gauss = (cl_precision *) calloc(cli->subdivisions, sizeof(cl_precision));
    cli->h_kronrod = (cl_precision *) calloc(cli->subdivisions, sizeof(cl_precision));
    cli->subinterval_errors = (cl_precision *) calloc(cli->subdivisions, sizeof(cl_precision));

    if (!cli->h_kronrod || !cli->h_gauss || !cli->subdivisions)
    {
        perror("[Stable-OpenCl] Host memory allocation failed.");
        return -1;
    }

    return 0;
}

double stable_clinteg_integrate(struct stable_clinteg* cli, double a, double b, double epsabs, double epsrel, unsigned short limit,
                   double *result, double *abserr, struct StableDistStruct* dist)
{
    cl_int err = 0;
    size_t work_threads, workgroup_size;
    struct stable_info h_args;


    // TODO: Create a "StableParams" structure that holds all of this parameters
    //      and use that instead of embedding all of them in the general StableDist
    //      structure, so I can avoid all this useless copying.
    h_args.beta_ = dist->beta_;
    h_args.k1 = dist->k1;
    h_args.xxipow = dist->xxipow;
    h_args.ibegin = a;
    h_args.iend = b;
    h_args.theta0_ = dist->theta0_;
    h_args.alfa = dist->alfa;
    h_args.alfainvalfa1 = dist->alfainvalfa1;

    h_args.subinterval_length = (b - a) / (double) cli->subdivisions;
    h_args.half_subint_length = h_args.subinterval_length / 2;
    h_args.threads_per_interval = cli->points_rule / 2 + 1 + 1; // Extra thread for sum.
    h_args.gauss_points = (cli->points_rule / 2 + 1) / 2;
    h_args.kronrod_points = cli->points_rule / 2;

    if(dist->ZONE == ALFA_1)
        h_args.integrand = PDF_ALPHA_EQ1;
    else
        h_args.integrand = PDF_ALPHA_NEQ1;

    stablecl_log(log_message, "[Stable-OpenCL] Integration begin - interval (%.3lf, %.3lf), %d subdivisions, %e subinterval length, %u threads per interval.\n", 
        a, b, cli->subdivisions, h_args.subinterval_length, h_args.threads_per_interval);

    cl_mem gauss = clCreateBuffer(cli->env.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                sizeof(cl_precision) * cli->subdivisions, NULL, &err);
    cl_mem kronrod = clCreateBuffer(cli->env.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                sizeof(cl_precision) * cli->subdivisions, NULL, &err);
    cl_mem args = clCreateBuffer(cli->env.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(struct stable_info)
                                 , &h_args, &err);

    if (!gauss || !kronrod || !args)
    {
        stablecl_log(log_err, "[Stable-OpenCl] Buffer creation failed with code %d: %s\n", err, opencl_strerr(err));
        stablecl_log(log_err, "[Stable-OpenCl] Pointers gauss, kronrod, args: %p, %p, %p\n", gauss, kronrod, args);
        goto cleanup;
    }

    int argc = 0;
    err |= clSetKernelArg(cli->env.kernel, argc++, sizeof(cl_mem), &gauss);
    err |= clSetKernelArg(cli->env.kernel, argc++, sizeof(cl_mem), &kronrod);
    err |= clSetKernelArg(cli->env.kernel, argc++, sizeof(cl_mem), &args);

    if (err)
    {
        stablecl_log(log_err, "[Stable-OpenCl] Couldn't set kernel arguments: error %d\n", err);
        goto cleanup;
    }

    workgroup_size = 64; // Minimum accepted number, it seems.
    work_threads = max(h_args.threads_per_interval, workgroup_size) * cli->subdivisions; // We already checked for overflow.

    stablecl_log(log_message, "[Stable-OpenCl] Enqueing kernel - %zu work threads, %zu workgroup size (%d points per interval)\n", work_threads, workgroup_size, cli->points_rule);

    err = clEnqueueNDRangeKernel(cli->env.queue, cli->env.kernel,
                                 1, NULL, &work_threads, &workgroup_size, 0, NULL, NULL);

    if (err)
    {
        stablecl_log(log_err, "[Stable-OpenCl] Error enqueueing the kernel command: %s (%d)\n", opencl_strerr(err), err);
        goto cleanup;
    }

    err |= clEnqueueReadBuffer(cli->env.queue, gauss, CL_TRUE, 0, sizeof(cl_precision) * cli->subdivisions,
                               cli->h_gauss, 0, NULL, NULL);
    err |= clEnqueueReadBuffer(cli->env.queue, kronrod, CL_TRUE, 0, sizeof(cl_precision) * cli->subdivisions,
                               cli->h_kronrod, 0, NULL, NULL);

    if (err)
    {
        stablecl_log(log_err, "[Stable-OpenCl] Error reading results from the GPU: %d\n", err);
        goto cleanup;
    }

    _stable_set_results(cli);

    *result = cli->result;
    *abserr = cli->abs_error;

cleanup:
    if (gauss) clReleaseMemObject(gauss);
    if (kronrod) clReleaseMemObject(kronrod);

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

    stablecl_log(log_message, "[Stable-OpenCl] Results set: gauss_sum = %.3g, kronrod_sum = %.3g\n", gauss_sum, kronrod_sum);

    cli->result = kronrod_sum;
    cli->abs_error = kronrod_sum - gauss_sum;

    return 0;
}

void stable_clinteg_teardown(struct stable_clinteg *cli)
{
    clReleaseKernel(cli->env.kernel);
    clReleaseProgram(cli->env.program);
    clReleaseCommandQueue(cli->env.queue);
    clReleaseContext(cli->env.context);
}


