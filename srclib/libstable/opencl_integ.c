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

static int _stable_create_points_array(struct stable_clinteg *cli, cl_precision *points, size_t num_points)
{
    int err;

    if (cli->points)
        clReleaseMemObject(cli->points);
    if (cli->gauss)
        clReleaseMemObject(cli->gauss);
    if (cli->kronrod)
        clReleaseMemObject(cli->kronrod);

    cli->points = clCreateBuffer(cli->env.context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                 sizeof(cl_precision) * num_points, points, &err);
    cli->gauss = clCreateBuffer(cli->env.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                 sizeof(cl_precision) * num_points, NULL, &err);
    cli->kronrod = clCreateBuffer(cli->env.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                 sizeof(cl_precision) * num_points, NULL, &err);

    return err;
}

static int _stable_map_gk_buffers(struct stable_clinteg *cli, size_t points)
{
    int err;

    cli->h_gauss = clEnqueueMapBuffer(cli->env.queue, cli->gauss, CL_TRUE, CL_MAP_READ, 0, points * sizeof(cl_precision), 0, NULL, NULL, &err);
    cli->h_kronrod = clEnqueueMapBuffer(cli->env.queue, cli->kronrod, CL_TRUE, CL_MAP_READ, 0, points * sizeof(cl_precision), 0, NULL, NULL, &err);

    return err;
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

    if (opencl_initenv(&cli->env, "opencl/stable_pdf.cl", "stable_pdf_points"))
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

    cli->gauss = NULL;
    cli->kronrod = NULL;
    cli->points = NULL;
    cli->args = clCreateBuffer(cli->env.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                               sizeof(struct stable_info), NULL, &err);

    if (err)
    {
        stablecl_log(log_err, "[Stable-OpenCl] Buffer creation failed: %s\n", opencl_strerr(err));
        return -1;
    }

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

short stable_clinteg_points(struct stable_clinteg *cli, double *x, double *pdf_results, double *errs, size_t num_points, struct StableDistStruct *dist)
{
    cl_int err = 0;
    size_t work_threads[2] = { KRONROD_EVAL_POINTS * num_points, GK_SUBDIVISIONS };
    size_t workgroup_size[2] = { KRONROD_EVAL_POINTS, GK_SUBDIVISIONS };
    cl_event event;

    cli->h_args->k1 = dist->k1;
    cli->h_args->alfa = dist->alfa;
    cli->h_args->alfainvalfa1 = dist->alfainvalfa1;

    if (dist->ZONE == GPU_TEST_INTEGRAND)
        cli->h_args->integrand = GPU_TEST_INTEGRAND;
    else if (dist->ZONE == GPU_TEST_INTEGRAND_SIMPLE)
        cli->h_args->integrand = GPU_TEST_INTEGRAND_SIMPLE;
    else if (dist->ZONE == ALFA_1)
        cli->h_args->integrand = PDF_ALPHA_EQ1;
    else
        cli->h_args->integrand = PDF_ALPHA_NEQ1;


    err |= clEnqueueWriteBuffer(cli->env.queue, cli->args, CL_FALSE, 0, sizeof(struct stable_info), cli->h_args, 0, NULL, NULL);
    err |= _stable_create_points_array(cli, x, num_points);

    if (err)
    {
        stablecl_log(log_err, "[Stable-OpenCl] Couldn't set buffers: %d (%s)\n", err, opencl_strerr(err));
        goto cleanup;
    }

    bench_begin(cli->profiling.argset, cli->profile_enabled);
    int argc = 0;
    err |= clSetKernelArg(cli->env.kernel, argc++, sizeof(cl_mem), &cli->args);
    err |= clSetKernelArg(cli->env.kernel, argc++, sizeof(cl_mem), &cli->points);
    err |= clSetKernelArg(cli->env.kernel, argc++, sizeof(cl_mem), &cli->gauss);
    err |= clSetKernelArg(cli->env.kernel, argc++, sizeof(cl_mem), &cli->kronrod);
    bench_end(cli->profiling.argset, cli->profile_enabled);

    if (err)
    {
        stablecl_log(log_err, "[Stable-OpenCl] Couldn't set kernel arguments: error %d\n", err);
        goto cleanup;
    }

    stablecl_log(log_message, "[Stable-OpenCl] Enqueing kernel - %zu × %zu work threads, %zu × %zu workgroup size\n", work_threads[0], work_threads[1], workgroup_size[0], workgroup_size[1], cli->points_rule);

    bench_begin(cli->profiling.enqueue, cli->profile_enabled);
    err = clEnqueueNDRangeKernel(cli->env.queue, cli->env.kernel,
                                 2, NULL, work_threads, workgroup_size, 0, NULL, &event);
    bench_end(cli->profiling.enqueue, cli->profile_enabled);

    if (err)
    {
        stablecl_log(log_err, "[Stable-OpenCl] Error enqueueing the kernel command: %s (%d)\n", opencl_strerr(err), err);
        goto cleanup;
    }

    bench_begin(cli->profiling.buffer_read, cli->profile_enabled);
    err = _stable_map_gk_buffers(cli, num_points);
    bench_end(cli->profiling.buffer_read, cli->profile_enabled);

    if (err)
    {
        stablecl_log(log_err, "[Stable-OpenCl] Error reading results from the GPU: %d\n", err);
        goto cleanup;
    }

    if (cli->profile_enabled)
        stable_retrieve_profileinfo(cli, event);

    bench_begin(cli->profiling.set_results, cli->profile_enabled);
    for(size_t i = 0; i < num_points; i++)
    {
        pdf_results[i] = cli->h_kronrod[i];
        errs[i] = cli->h_kronrod[i] - cli->h_gauss[i];
        stablecl_log(log_message, "[Stable-OpenCl] Results set P%zu: gauss_sum = %.3g, kronrod_sum = %.3g, abserr = %.3g\n", i, cli->h_gauss[i], cli->h_kronrod[i], errs[i]);
    }
    bench_end(cli->profiling.set_results, cli->profile_enabled);


cleanup:
    stablecl_log(log_message, "[Stable-OpenCl] Integration end.\n");

    return err;
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


