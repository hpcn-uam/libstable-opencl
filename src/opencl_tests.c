#include <stdio.h>
#include <time.h>
#include <libgen.h>
#include <math.h>

#include "openclenv.h"
#include "opencl_common.h"
#define testtype int
#define max(a,b) ((a) < (b) ? (b) : (a))

short test_instance(struct openclenv* ocl, size_t size, size_t dim,
	const size_t* global_work_size, const size_t* local_work_size, struct opencl_profile* profiling)
{
	testtype* array;
	cl_mem array_ocl;
	cl_int err;
	cl_event event;
    testtype sum = 0;

 	array = (testtype *) calloc(size, sizeof(testtype));

    if (!array)
    {
        perror("[Stable-OpenCl] Host memory allocation failed.");
        return -1;
    }

    for(size_t i = 0; i < size; i++)
    {
        array[i] = rand() & 0xFF;
        sum += array[i];
    }

    array_ocl = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                sizeof(testtype) * size, array, &err);
    if (err)
    {
        stablecl_log(log_err, "Buffer creation failed: %s", opencl_strerr(err));
        goto cleanup;
    }

    int argc = 0;
    err |= clSetKernelArg(ocl->kernel[0], argc++, sizeof(cl_mem), &array_ocl);
    err |= clSetKernelArg(ocl->kernel[0], argc++, sizeof(testtype) * (*local_work_size), NULL);
    err = clEnqueueNDRangeKernel(opencl_get_queue(ocl), ocl->kernel[0],
                                dim, NULL, global_work_size, local_work_size, 0, NULL, &event);

    if(err)
    {
        stablecl_log(log_err, "Error enqueueing the kernel command: %s (%d).", opencl_strerr(err), err);
        goto cleanup;
    }

    stablecl_finish_all(ocl);
    stablecl_profileinfo(profiling, event);

    if(array[0] != sum)
        stablecl_log(log_err, "Error: expected result is %ld, actual was %ld.", sum, array[0]);

cleanup:
    if(array_ocl)
        clEnqueueUnmapMemObject(opencl_get_queue(ocl), array_ocl, array, 0, NULL, NULL);

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
	size_t workgroup_sizes[] = { 64, 128, 256, 512, 1024 };
	size_t array_size;
	size_t array_size_tests = 25;
    double bw;
    size_t array_bytes;
	int wg_i, as_i;

	snprintf(profile_fname, 100, "%s.dat", kernel_name);

	profile_f = fopen(profile_fname, "w");

	if (opencl_initenv(&ocl) || opencl_load_kernel(&ocl, file, kernel_name, 0))
    {
        stablecl_log(log_message, "OpenCL environment failure.");
        return -1;
    }

    stablecl_log(log_message, "Testing kernel %s...", kernel_name);

    for(wg_i = 0; wg_i < sizeof workgroup_sizes / sizeof(size_t); wg_i++)
    {
    	for(as_i = 6; as_i < array_size_tests; as_i++)
    	{
    		array_size = 1 << as_i;
            array_size = max(array_size, workgroup_sizes[wg_i]);
            test_instance(&ocl, array_size, 1, &array_size, workgroup_sizes + wg_i, &profiling);

            array_bytes = array_size * sizeof(testtype);
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
    // test_kernel("opencl/perftests.cl", "array_sum_reduction");
    test_kernel("opencl/perftests.cl", "array_sum_twostage_loop");
    test_kernel("opencl/perftests.cl", "array_sum_twostage_loop_lc");
    test_kernel("opencl/perftests.cl", "array_sum_twostage_reduction");
    test_kernel("opencl/perftests.cl", "array_sum_twostage_half_wgs");
    // test_kernel("opencl/perftests.cl", "array_sum_twostage_two_wgs");
    // test_kernel("opencl/perftests.cl", "array_sum_2stage_lc");

	return 0;
}
