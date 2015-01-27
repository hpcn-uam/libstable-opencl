#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#warning "Double precision floating point not supported by OpenCL implementation."
#endif

#include "includes/opencl_common.h"

kernel void array_sum_loop(global cl_precision* array)
{
	size_t local_wg_index = get_local_id(0);
	size_t group_index = get_group_id(0);
	size_t array_size = get_global_size(0);
	size_t global_index = get_global_id(0);

	if(global_index == 0)
	{
		cl_precision sum = 0;

		for(int i = 0; i < array_size; i++)
		{
			sum += array[i];
		}

		array[0] = sum;
	}

	barrier(CLK_GLOBAL_MEM_FENCE);
}

kernel void array_sum_reduction(global cl_precision* array)
{
	size_t local_wg_index = get_local_id(0);
	size_t group_index = get_group_id(0);
	size_t array_size = get_global_size(0);
	size_t global_index = get_global_id(0);

	barrier(CLK_GLOBAL_MEM_FENCE);

	for(size_t offset = array_size / 2; offset > 0; offset >>= 1)
	{
	    if (global_index < offset)
	    	array[global_index] += array[global_index + offset];

	    barrier(CLK_GLOBAL_MEM_FENCE);
	}
}
