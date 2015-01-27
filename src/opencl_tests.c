#include <stdio.h>
#include <time.h>
#include <libgen.h>
#include <math.h>

#include "openclenv.h"
#include "opencl_common.h"

#define max(a,b) ((a) < (b) ? (b) : (a))

short test_instance(struct openclenv* ocl, size_t size, size_t dim,
	const size_t* global_work_size, const size_t* local_work_size, struct opencl_profile* profiling)
{
	long* array;
	cl_mem array_ocl;
	cl_int err;
	cl_event event;
    long sum = 0;

 	array = (long *) calloc(size, sizeof(long));

    if (!array)
    {
        perror("[Stable-OpenCl] Host memory allocation failed.");
        return -1;
    }

    for(size_t i = 0; i < size; i++)
    {
        array[i] = 10 * ((long) rand() / (long) RAND_MAX);
        sum += array[i];
    }

    array_ocl = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                sizeof(long) * size, array, &err);
    if (err)
    {
        stablecl_log(log_err, "[Stable-OpenCl] Buffer creation failed: %s\n", opencl_strerr(err));
        goto cleanup;
    }

    int argc = 0;
    err |= clSetKernelArg(ocl->kernel, argc++, sizeof(cl_mem), &array_ocl);
    err = clEnqueueNDRangeKernel(ocl->queue, ocl->kernel,
                                dim, NULL, global_work_size, local_work_size, 0, NULL, &event);

    if(err)
    {
        stablecl_log(log_err, "[Stable-OpenCl] Error enqueueing the kernel command: %s (%d)\n", opencl_strerr(err), err);
        goto cleanup;
    }

    stablecl_finish_all(ocl);
    stablecl_profileinfo(profiling, event);

    if(fabs(array[0] - sum) > 1)
        stablecl_log(log_err, "[Stable-OpenCl] Error: expected result is %.3lf, actual was %.3lf\n", sum, array[0]);

cleanup:
    if(array_ocl)
        clEnqueueUnmapMemObject(ocl->queue, array_ocl, array, 0, NULL, NULL);

    if(array)
        free(array);

    return 0;
}

short test_kernel(const char* file, const char* kernel_name)
{
	struct openclenv ocl;
	struct opencl_profile profiling;
	size_t global_size;
	char profile_fname[100];
	FILE* profile_f;
	size_t workgroup_sizes[] = { 64, 128, 256, 512 };
	size_t array_size;
	size_t array_size_tests = 23;
    double bw;
    size_t array_bytes;
	int wg_i, as_i;

	snprintf(profile_fname, 100, "%s.dat", kernel_name);

	profile_f = fopen(profile_fname, "w");

	if (opencl_initenv(&ocl, file, kernel_name))
    {
        stablecl_log(log_message, "[Stable-OpenCl] OpenCL environment failure.\n");
        return -1;
    }

    stablecl_log(log_message, "[Stable-OpenCl] Testing kernel %s\n", kernel_name);

    for(wg_i = 0; wg_i < sizeof workgroup_sizes / sizeof(size_t); wg_i++)
    {
    	for(as_i = 6; as_i < array_size_tests; as_i++)
    	{
    		array_size = 1 << as_i;
            array_size = max(array_size, workgroup_sizes[wg_i]);
            test_instance(&ocl, array_size, 1, &array_size, workgroup_sizes + wg_i, &profiling);

            array_bytes = array_size * sizeof(long);
            bw = array_bytes / profiling.exec_time;
            fprintf(profile_f, "%zu\t%zu\t%.3lf\t%.3lf\n", array_size, workgroup_sizes[wg_i],
                profiling.exec_time, 8 * bw / (1024 * 1024 * 1024));
    	}
    }

    opencl_teardown(&ocl);
    fclose(profile_f);

    return 0;
}

int main(int argc, char const *argv[])
{
	srand(time(NULL));

    // test_kernel("opencl/perftests.cl", "array_sum_loop");
    test_kernel("opencl/perftests.cl", "array_sum_reduction");
    test_kernel("opencl/perftests.cl", "array_sum_twostage_loop");
    test_kernel("opencl/perftests.cl", "array_sum_twostage_reduction");
    test_kernel("opencl/perftests.cl", "array_sum_twostage_half_wgs");
    test_kernel("opencl/perftests.cl", "array_sum_twostage_two_wgs");

	return 0;
}
