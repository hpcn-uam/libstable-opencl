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

kernel void array_sum_twostage_loop(global cl_precision* array)
{
	size_t local_wg_index = get_local_id(0);
	size_t group_index = get_group_id(0);
	size_t array_size = get_global_size(0);
	size_t global_index = get_global_id(0);
	size_t wg_size = get_local_size(0);
	size_t local_offset = group_index * wg_size;
	size_t group_count = get_num_groups(0);

	barrier(CLK_LOCAL_MEM_FENCE);

	for(size_t offset = wg_size / 2; offset > 0; offset >>= 1)
	{
	    if (local_wg_index < offset)
	    	array[local_offset + local_wg_index] += array[local_offset + local_wg_index + offset];

	    barrier(CLK_LOCAL_MEM_FENCE);
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if(global_index == 0)
	{
		cl_precision sum = 0;
		for(size_t i = 0; i < array_size; i += wg_size)
		{
			sum += array[i];
		}

		array[0] = sum;
	}
}



kernel void array_sum_twostage_reduction(global cl_precision* array)
{
	size_t local_wg_index = get_local_id(0);
	size_t group_index = get_group_id(0);
	size_t array_size = get_global_size(0);
	size_t global_index = get_global_id(0);
	size_t wg_size = get_local_size(0);
	size_t local_offset = group_index * wg_size;
	size_t group_count = get_num_groups(0);

	barrier(CLK_LOCAL_MEM_FENCE);

	for(size_t offset = wg_size / 2; offset > 0; offset >>= 1)
	{
	    if (local_wg_index < offset)
	    	array[local_offset + local_wg_index] += array[local_offset + local_wg_index + offset];

	    barrier(CLK_LOCAL_MEM_FENCE);
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	for(size_t offset = group_count / 2; offset > 0; offset >>= 1)
	{
	    if (local_wg_index == 0 && group_index < offset)
	    	array[group_index] += array[group_index + offset];

	    barrier(CLK_GLOBAL_MEM_FENCE);
	}

}
