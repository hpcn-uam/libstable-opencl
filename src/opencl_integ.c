#include "opencl_integ.h"

#include <limits.h>
#include <stdio.h>


struct stable_info {
    double beta_;
    double k1;
    double xxipow;
    double ibegin;
    double iend;
};

static int _stable_set_results(struct stable_clinteg *cli);

static int _stable_can_overflow(struct stable_clinteg *cli)
{
    cl_uint work_threads = cli->points_rule * cli->subdivisions;

    return work_threads / cli->subdivisions != cli->points_rule;
}

int stable_clinteg_init(struct stable_clinteg *cli)
{
    cli->points_rule = 61;
    cli->subdivisions = 201;

    if (_stable_can_overflow(cli))
    {
        fprintf(stderr, "[Stable-OpenCl] Warning: possible overflow in work dimension (%d x %d).\n"
                , cli->points_rule, cli->subdivisions);
        return -1;
    }

    if (opencl_initenv(&cli->env, "obj/stable_pdf.bc", "stable_pdf"))
    {
        fprintf(stderr, "[Stable-OpenCl] OpenCL environment failure.\n");
        return -1;
    }

    cli->h_gauss = (double *) calloc(cli->subdivisions, sizeof(double));
    cli->h_kronrod = (double *) calloc(cli->subdivisions, sizeof(double));
    cli->subinterval_errors = (double *) calloc(cli->subdivisions, sizeof(double));

    if (!cli->h_kronrod || !cli->h_gauss || !cli->subdivisions)
    {
        perror("[Stable-OpenCl] Host memory allocation failed.");
        return -1;
    }

    return 0;
}

double stable_clinteg_integrate(struct stable_clinteg* cli, double a, double b, double epsabs, double epsrel, unsigned short limit,
                   double *result, double *abserr, double beta_, double k1, double xxipow)
{
    cl_int err = 0;
    size_t work_threads;
    struct stable_info h_args;

    h_args.beta_ = beta_;
    h_args.k1 = k1;
    h_args.xxipow = xxipow;
    h_args.ibegin = a;
    h_args.iend = b;

    cl_mem gauss = clCreateBuffer(cli->env.context, CL_MEM_WRITE_ONLY,
                                  sizeof(double) * cli->subdivisions, NULL, &err);
    cl_mem kronrod = clCreateBuffer(cli->env.context, CL_MEM_WRITE_ONLY,
                                    sizeof(double) * cli->subdivisions, NULL, &err);
    cl_mem args = clCreateBuffer(cli->env.context, CL_MEM_READ_ONLY, sizeof(struct stable_info)
                                   , &h_args, &err);

    if (!gauss || !kronrod || !args)
    {
        fprintf(stderr, "[Stable-OpenCl] Buffer creation failed with code %d\n", err);
        goto cleanup;
    }

    int argc = 0;
    err |= clSetKernelArg(cli->env.kernel, argc++, sizeof(cl_mem), &gauss);
    err |= clSetKernelArg(cli->env.kernel, argc++, sizeof(cl_mem), &kronrod);
    err |= clSetKernelArg(cli->env.kernel, argc++, sizeof(cl_mem), &args);

    if (err)
    {
        fprintf(stderr, "[Stable-OpenCl] Couldn't set kernel arguments: error %d\n", err);
        goto cleanup;
    }

    work_threads = cli->points_rule * cli->subdivisions; // We already checked for overflow.

    err = clEnqueueNDRangeKernel(cli->env.queue, cli->env.kernel,
                                 1, NULL, &work_threads, NULL, 0, NULL, NULL);

    if (err)
    {
        fprintf(stderr, "[Stable-OpenCl] Error enqueueing the kernel command: %d\n", err);
        goto cleanup;
    }

    err |= clEnqueueReadBuffer(cli->env.queue, gauss, CL_TRUE, 0, sizeof(double) * cli->subdivisions, 
        cli->h_gauss, 0, NULL, NULL);
    err |= clEnqueueReadBuffer(cli->env.queue, kronrod, CL_TRUE, 0, sizeof(double) * cli->subdivisions, 
        cli->h_kronrod, 0, NULL, NULL);

    if (err)
    {
        fprintf(stderr, "[Stable-OpenCl] Error reading results from the GPU: %d\n", err);
        goto cleanup;
    }

    _stable_set_results(cli);

    *result = cli->result;
    *abserr = cli->abs_error;

cleanup:
    if (gauss) clReleaseMemObject(gauss);
    if (kronrod) clReleaseMemObject(kronrod);

    return err;
}

static int _stable_set_results(struct stable_clinteg *cli)
{
    double gauss_sum = 0, kronrod_sum = 0;
    unsigned int i;

    for (i = 0; i < cli->subdivisions; i++)
    {
        gauss_sum += cli->h_gauss[i];
        kronrod_sum += cli->h_kronrod[i];
        cli->subinterval_errors[i] = cli->h_gauss[i] - cli->h_kronrod[i];
    }

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


