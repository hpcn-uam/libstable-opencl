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

#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#warning "Double precision floating point not supported by OpenCL implementation."
#endif

#include "includes/opencl_common.h"
#define testtype int

kernel void array_sum_loop(global testtype* array, local testtype* scratch)
{
	size_t local_wg_index = get_local_id(0);
	size_t group_index = get_group_id(0);
	size_t array_size = get_global_size(0);
	size_t global_index = get_global_id(0);

	if(global_index == 0)
	{
		testtype sum = 0;

		for(int i = 0; i < array_size; i++)
		{
			sum += array[i];
		}

		array[0] = sum;
	}

	barrier(CLK_GLOBAL_MEM_FENCE);
}

kernel void array_sum_reduction(global testtype* array, local testtype* scratch)
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

kernel void array_sum_twostage_loop(global testtype* array, local testtype* scratch)
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
		testtype sum = 0;
		for(size_t i = 0; i < array_size; i += wg_size)
		{
			sum += array[i];
		}

		array[0] = sum;
	}
}

kernel void array_sum_twostage_reduction(global testtype* array, local testtype* scratch)
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

kernel void array_sum_twostage_two_wgs(global testtype* array, local testtype* scratch)
{
	size_t local_wg_index = get_local_id(0);
	size_t group_index = get_group_id(0);
	size_t array_size = get_global_size(0);
	size_t global_index = get_global_id(0);
	size_t wg_size = get_local_size(0);
	size_t local_offset;
	size_t group_count = get_num_groups(0);
	size_t actual_group_count = 2;
	testtype sum;

	barrier(CLK_LOCAL_MEM_FENCE);

	if(group_index < actual_group_count)
	{
		for(size_t chunk_index = group_index; chunk_index < group_count; chunk_index += actual_group_count)
		{
			local_offset = chunk_index * wg_size;

			for(size_t offset = wg_size / 2; offset > 0; offset >>= 1)
			{
			    if (local_wg_index < offset)
			    	array[local_offset + local_wg_index] += array[local_offset + local_wg_index + offset];

			    barrier(CLK_LOCAL_MEM_FENCE);
			}

			if(local_wg_index == 0 && chunk_index != group_index)
				array[group_index * wg_size] += array[local_offset];

			barrier(CLK_LOCAL_MEM_FENCE);
		}

		barrier(CLK_GLOBAL_MEM_FENCE);

		if(group_index == 0 && local_wg_index == 0)
			array[0] += array[wg_size];
	}
}

kernel void array_sum_twostage_half_wgs(global testtype* array, local testtype* scratch)
{
	size_t local_wg_index = get_local_id(0);
	size_t group_index = get_group_id(0);
	size_t array_size = get_global_size(0);
	size_t global_index = get_global_id(0);
	size_t wg_size = get_local_size(0);
	size_t local_offset;
	size_t group_count = get_num_groups(0);
	size_t actual_group_count = group_count / 2;
	size_t final_reduction_needed_groups = actual_group_count / wg_size;
	testtype sum;

	if(actual_group_count % wg_size != 0)
		final_reduction_needed_groups += 1;

	barrier(CLK_LOCAL_MEM_FENCE);

	if(group_index < actual_group_count)
	{
		for(size_t chunk_index = group_index; chunk_index < group_count; chunk_index += actual_group_count)
		{
			local_offset = chunk_index * wg_size;

			for(size_t offset = wg_size / 2; offset > 0; offset >>= 1)
			{
			    if (local_wg_index < offset)
			    	array[local_offset + local_wg_index] += array[local_offset + local_wg_index + offset];

			    barrier(CLK_LOCAL_MEM_FENCE);
			}

			if(local_wg_index == 0)
				array[group_index] += array[local_offset];

			barrier(CLK_LOCAL_MEM_FENCE);
		}
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	while(final_reduction_needed_groups > 0)
	{
		if(group_index < final_reduction_needed_groups)
		{
			size_t actual_reduction_size = wg_size;

			if(group_index == final_reduction_needed_groups - 1)
				actual_reduction_size = actual_reduction_size - final_reduction_needed_groups * wg_size;

			for(size_t offset = wg_size / 2; offset > 0; offset >>= 1)
			{
				if(local_wg_index < offset && local_wg_index + offset < actual_reduction_size)
					array[local_wg_index] += array[(local_wg_index + group_index) * wg_size];

				barrier(CLK_LOCAL_MEM_FENCE);
			}
		}

		if(final_reduction_needed_groups > 1)
			final_reduction_needed_groups /= wg_size;
		else
			final_reduction_needed_groups = 0;

		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

kernel void array_sum_2stage_lc(global testtype* array, local volatile testtype* sdata)
{
    size_t local_wg_index = get_local_id(0);
	size_t group_index = get_group_id(0);
	size_t array_size = get_global_size(0);
	size_t global_index = get_global_id(0);
	size_t wg_size = get_local_size(0);
	size_t local_offset = group_index * wg_size;
	size_t group_count = get_num_groups(0);
	size_t offset;

	barrier(CLK_LOCAL_MEM_FENCE);

	// Unroll and to local
	offset = wg_size / 2;

	if (local_wg_index < offset)
    	sdata[local_wg_index] = array[local_offset + local_wg_index + offset] + array[local_offset + local_wg_index];

    offset >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE);

	for(; offset > 0; offset >>= 1)
	{
	    if (local_wg_index < offset)
	    	sdata[local_wg_index] += sdata[local_wg_index + offset];

	    barrier(CLK_LOCAL_MEM_FENCE);
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	for(offset = group_count / 2; offset > 0; offset >>= 1)
	{
	    if (local_wg_index == 0 && group_index < offset)
	    	array[group_index] += array[group_index + offset];

	    barrier(CLK_GLOBAL_MEM_FENCE);
	}
}


kernel void array_sum_twostage_loop_lc(global testtype* array, local testtype* sdata)
{
	size_t local_wg_index = get_local_id(0);
	size_t group_index = get_group_id(0);
	size_t array_size = get_global_size(0);
	size_t global_index = get_global_id(0);
	size_t wg_size = get_local_size(0);
	size_t local_offset = group_index * wg_size;
	size_t group_count = get_num_groups(0);
size_t offset;

	barrier(CLK_LOCAL_MEM_FENCE);

	// Unroll and to local
	offset = wg_size / 2;

	if (local_wg_index < offset)
    	sdata[local_wg_index] = array[local_offset + local_wg_index + offset] + array[local_offset + local_wg_index];

    offset >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE);

	for(; offset > 0; offset >>= 1)
	{
	    if (local_wg_index < offset)
	    	sdata[local_wg_index] += sdata[local_wg_index + offset];

	    barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(local_wg_index == 0)
		array[group_index] = sdata[0];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if(global_index == 0)
	{
		testtype sum = 0;
		for(size_t i = 0; i < array_size; i += wg_size)
		{
			sum += array[i];
		}

		array[0] = sum;
	}
}
