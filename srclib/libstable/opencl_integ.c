/*
 * Copyright (C) 2015 - Naudit High Performance Computing and Networking
 *
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

#include "stable_api.h"
#include "opencl_common.h"
#include "benchmarking.h"

#include <limits.h>
#include <stdio.h>
#include <stdarg.h>

#ifndef max
#define max(a,b) (a < b ? b : a)
#endif

#define KERNIDX_INTEGRATE 0
#define KERNIDX_QUANTILE 2
#define KERNIDX_RNG 3
#define KERN_POINTS_NAME "stable_points"
#define KERN_QUANTILE_NAME "stable_quantile"
#define KERN_RNG_NAME "stable_rng"

#define MIN_POINTS_PER_QUEUE 200

static int _stable_can_overflow(struct stable_clinteg *cli)
{
	cl_uint work_threads = cli->points_rule * cli->subdivisions;

	return work_threads / cli->subdivisions != cli->points_rule;
}

static size_t _stable_get_maximum_points_for_gpu(struct stable_clinteg* cli)
{
	size_t constant_memory_per_point = 0;
	size_t global_memory_per_point = 0;
	size_t max_points_constant, max_points_global;

	if (!cli->mode_pointgenerator) {
		constant_memory_per_point = sizeof(cl_precision);
		global_memory_per_point = sizeof(cl_precision);
	}

	global_memory_per_point += sizeof(cl_precision);

	max_points_constant = cli->env.max_constant_memory / constant_memory_per_point;
	max_points_global = cli->env.max_global_memory / global_memory_per_point;

	if (max_points_global > max_points_constant)
		return max_points_constant;
	else
		return max_points_global;
}

static int _stable_create_points_array(struct stable_clinteg *cli, cl_precision *points, size_t num_points)
{
	int err = 0;

	if (cli->points)
		clReleaseMemObject(cli->points);

	if (cli->gauss)
		clReleaseMemObject(cli->gauss);

	if (cli->kronrod)
		clReleaseMemObject(cli->kronrod);

	if (!cli->mode_pointgenerator)
		cli->points = clCreateBuffer(cli->env.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
									 sizeof(cl_precision) * num_points, points, &err);

	if (err) return err;

	if (!cli->mode_pointgenerator)
		cli->gauss = clCreateBuffer(cli->env.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
									sizeof(cl_precision) * num_points, NULL, &err);

	if (err) return err;

	cli->kronrod = clCreateBuffer(cli->env.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
								  sizeof(cl_precision) * num_points, NULL, &err);

	stablecl_log(log_message, "Size sent is %zu\n", sizeof(cl_precision) * num_points);

	return err;
}

static int _stable_map_gk_buffers(struct stable_clinteg *cli, size_t points)
{
	int err = 0;

	if (!cli->mode_pointgenerator)
		cli->h_gauss = clEnqueueMapBuffer(opencl_get_queue(&cli->env), cli->gauss, CL_FALSE, CL_MAP_READ, 0, points * sizeof(cl_precision), 0, NULL, NULL, &err);

	cli->h_kronrod = clEnqueueMapBuffer(opencl_get_queue(&cli->env), cli->kronrod, CL_TRUE, CL_MAP_READ, 0, points * sizeof(cl_precision), 0, NULL, NULL, &err);

	return err;
}

static int  _stable_unmap_gk_buffers(struct stable_clinteg* cli)
{
#ifndef SIMULATOR_BUILD
	int err = 0;

	if (!cli->mode_pointgenerator)
		err = clEnqueueUnmapMemObject(opencl_get_queue(&cli->env), cli->gauss, cli->h_gauss, 0, NULL, NULL);

	if (err) return err;

	cli->h_gauss = NULL;

	err = clEnqueueUnmapMemObject(opencl_get_queue(&cli->env), cli->kronrod, cli->h_kronrod, 0, NULL, NULL);

	if (err) return err;

	cli->h_kronrod = NULL;
#endif

	return 0;
}

static int _stable_clinteg_load_kernels(struct openclenv* env)
{
	char* kern_names[] = { KERN_POINTS_NAME, KERN_QUANTILE_NAME, KERN_RNG_NAME };
	size_t kern_indexes[] = { KERNIDX_INTEGRATE, KERNIDX_QUANTILE, KERNIDX_RNG };
	size_t kern_count = sizeof(kern_indexes) / sizeof(size_t);
	size_t i;

	for (i = 0; i < kern_count; i++) {
		if (opencl_load_kernel(env, "opencl/stable.cl", kern_names[i], kern_indexes[i])) {
			stablecl_log(log_err, "OpenCL kernel %s load failure.", kern_names[i]);
			return -1;
		} else
			stablecl_log(log_message, "OpenCL kernel %s loaded with index %zu.", kern_names[i], kern_indexes[i]);
	}

	return 0;
}

int stable_clinteg_init(struct stable_clinteg *cli, size_t platform_index)
{
	int err;

	cli->points_rule = GK_POINTS;
	cli->subdivisions = GK_SUBDIVISIONS;

#ifdef BENCHMARK
	cli->profile_enabled = 1;
#else
	cli->profile_enabled = 0;
#endif

	if (_stable_can_overflow(cli)) {
		stablecl_log(log_warning, "Warning: possible overflow in work dimension (%d x %d)."
					 , cli->points_rule, cli->subdivisions);
		return -1;
	}

	if (opencl_initenv(&cli->env, platform_index)) {
		stablecl_log(log_message, "OpenCL environment failure.");
		return -1;
	}

	if (_stable_clinteg_load_kernels(&cli->env)) {
		stablecl_log(log_err, "Cannot load kernels.");
		return -1;
	}

	cli->subinterval_errors = (cl_precision *) calloc(cli->subdivisions, sizeof(cl_precision));

	if (!cli->subdivisions) {
		perror("Host memory allocation failed.");
		return -1;
	}

	cli->gauss = NULL;
	cli->kronrod = NULL;
	cli->points = NULL;
	cli->args = clCreateBuffer(cli->env.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
							   sizeof(struct stable_info), NULL, &err);

	if (err) {
		stablecl_log(log_err, "Buffer creation failed: %s", opencl_strerr(err));
		return -1;
	}

	cli->h_args = clEnqueueMapBuffer(opencl_get_queue(&cli->env), cli->args, CL_TRUE, CL_MAP_WRITE, 0, sizeof(struct stable_info), 0, NULL, NULL, &err);

	if (err) {
		stablecl_log(log_err, "Buffer mapping failed: %s. "
					 "Host pointers (gauss, kronrod, args): (%p, %p, %p)",
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

static void _stable_clinteg_prepare_kernel_data(struct stable_info* info, StableDist* dist)
{
	info->k1 = dist->k1;
	info->alfa = dist->alfa;
	info->alfainvalfa1 = dist->alfainvalfa1;
	info->beta = dist->beta;
	info->THETA_TH = stable_get_THETA_TH();
	info->theta0 = dist->theta0;
	info->xi = dist->xi;
	info->mu_0 = dist->mu_0;
	info->sigma = dist->sigma;
	info->xxi_th = stable_get_XXI_TH();
	info->c2_part = dist->c2_part;
	info->xi_coef = (exp(lgamma(1 + 1 / dist->alfa))) / (M_PI * pow(1 + dist->xi * dist->xi, 1 / (2 * dist->alfa)));
	info->c1 = dist->c1;
	info->final_pdf_factor = dist->c2_part / dist->sigma;
	info->final_cdf_factor = dist->alfa < 1 ? M_1_PI : - M_1_PI;
	info->final_cdf_addition = dist->c1;
	info->quantile_tolerance = 1e-4;

	if (dist->cli.mode_pointgenerator) {
		info->rng_seed_a = gsl_rng_uniform(dist->gslrand) * UINT32_MAX;
		info->rng_seed_b = gsl_rng_uniform(dist->gslrand) * UINT32_MAX;
		info->mu_0 = dist->mu_1;
		stablecl_log(log_message, "Random seeds: %u, %u\n", info->rng_seed_a, info->rng_seed_b);
	}

	if (dist->cli.mode_bits == MODEMARKER_PDF)
		info->max_reevaluations = dist->alfa > 1 ? 2 : 1;
	else
		info->max_reevaluations = 1;

	short alfa_marker = dist->ZONE == ALFA_1 ? MODEMARKER_EQ1 : MODEMARKER_NEQ1;
	info->integrand = alfa_marker | dist->cli.mode_bits;

	if (dist->ZONE == ALFA_1)
		info->beta = fabs(dist->beta);
}

cl_precision* _stable_check_precision_type(double* values, size_t num_points)
{
	cl_precision* points = (cl_precision*) values;

#ifdef CL_PRECISION_IS_FLOAT
	stablecl_log(log_message, "Using floats, forcing cast.");

	points = (cl_precision*) calloc(num_points, sizeof(cl_precision));

	if (!points) {
		stablecl_log(log_err, "Couldn't allocate memory.");
		return NULL;
	}

	for (size_t i = 0; i < num_points; i++)
		points[i] = (cl_precision) values[i];

#endif

	return points;
}

short stable_clinteg_points_async(struct stable_clinteg *cli, double *x, size_t num_points, struct StableDistStruct *dist, cl_event* event)
{
	cl_int err = 0;
	size_t regular_work_threads[2] = { KRONROD_EVAL_POINTS * num_points, MAX_WORKGROUPS };
	size_t regular_workgroup_size[2] = { KRONROD_EVAL_POINTS, MAX_WORKGROUPS };
	size_t pointgen_work_threads = num_points;
	size_t* work_threads, *workgroup_size, dimensions;
	size_t max_points = _stable_get_maximum_points_for_gpu(cli);
	cl_precision* points = NULL;

	if (num_points > max_points) {
		stablecl_log(log_warning, "Warning: calling with %zu points, greater than maximum supported by GPU (%zu points)",
					 num_points, max_points);
	}

	_stable_clinteg_prepare_kernel_data(cli->h_args, dist);

	if (!cli->mode_pointgenerator) {
		points = _stable_check_precision_type(x, num_points);

		if (!points)
			goto cleanup;
	}

	err |= clEnqueueWriteBuffer(opencl_get_queue(&cli->env), cli->args, CL_FALSE, 0, sizeof(struct stable_info), cli->h_args, 0, NULL, NULL);
	err |= _stable_create_points_array(cli, points, num_points);

	if (err) {
		stablecl_log(log_err, "Couldn't set buffers: %d (%s)", err, opencl_strerr(err));
		goto cleanup;
	}

	opencl_set_current_kernel(&cli->env, cli->kern_index);

	bench_begin(cli->profiling.argset, cli->profile_enabled);
	int argc = 0;
	err |= clSetKernelArg(opencl_get_current_kernel(&cli->env), argc++, sizeof(cl_mem), &cli->args);

	if (!cli->mode_pointgenerator) {
		err |= clSetKernelArg(opencl_get_current_kernel(&cli->env), argc++, sizeof(cl_mem), &cli->points);
		err |= clSetKernelArg(opencl_get_current_kernel(&cli->env), argc++, sizeof(cl_mem), &cli->gauss);
	}

	err |= clSetKernelArg(opencl_get_current_kernel(&cli->env), argc++, sizeof(cl_mem), &cli->kronrod);
	bench_end(cli->profiling.argset, cli->profile_enabled);

	if (err) {
		stablecl_log(log_err, "Couldn't set kernel arguments: error %d", err);
		goto cleanup;
	}

	if (cli->mode_pointgenerator) {
		work_threads = &pointgen_work_threads;
		workgroup_size = NULL; // Let the driver decide.
		dimensions = 1;

		stablecl_log(log_message, "Enqueing p.gen. kernel %d - %zu work threads", cli->kern_index, work_threads[0]);
	} else {
		work_threads = regular_work_threads;
		workgroup_size = regular_workgroup_size;
		dimensions = 2;

		stablecl_log(log_message, "Enqueing kernel %d - %zu × %zu work threads, %zu × %zu workgroup size",
					 cli->kern_index, work_threads[0], work_threads[1], workgroup_size[0], workgroup_size[1], cli->points_rule);
	}


	bench_begin(cli->profiling.enqueue, cli->profile_enabled);
	err = clEnqueueNDRangeKernel(opencl_get_queue(&cli->env), opencl_get_current_kernel(&cli->env),
								 dimensions, NULL, work_threads, workgroup_size, 0, NULL, event);
	bench_end(cli->profiling.enqueue, cli->profile_enabled);

	if (err) {
		stablecl_log(log_err, "Error enqueueing the kernel command: %s (%d)", opencl_strerr(err), err);
		goto cleanup;
	}

cleanup:
	stablecl_log(log_message, "Async command issued.");

#ifdef CL_PRECISION_IS_FLOAT

	if (points) free(points);

#endif

	return err;
}

short stable_clinteg_points(struct stable_clinteg *cli, double *x, double *results_1, double *results_2, double *errs, size_t num_points, struct StableDistStruct *dist)
{
	cl_event event;
	cl_int err;
	size_t max_points = _stable_get_maximum_points_for_gpu(cli);

	if (num_points > max_points) {
		// If the user requests more points than the number supported by GPU,
		// do it in two calls.
		stablecl_log(log_message, "num_points > max_points (%zu > %zu), separating in two calls", num_points, max_points);
		err = stable_clinteg_points(cli, x + max_points,
									results_1 != NULL ? results_1 + max_points : NULL,
									results_2 != NULL ? results_2 + max_points : NULL,
									errs != NULL ? errs + max_points : NULL,
									num_points - max_points, dist);
		num_points = max_points;

		if (err) return err;

	}

	err = stable_clinteg_points_async(cli, x, num_points, dist, &event);

	if (err) {
		stablecl_log(log_err, "Couldn't issue evaluation command to the GPU.");
		return err;
	}

	return stable_clinteg_points_end(cli, results_1, results_2, errs, num_points, dist, &event);
}

short stable_clinteg_points_end(struct stable_clinteg *cli, double *results_1, double *results_2, double* errs, size_t num_points, struct StableDistStruct *dist, cl_event* event)
{
	cl_int err = 0;

	if (event)
		clWaitForEvents(1, event);

	bench_begin(cli->profiling.buffer_read, cli->profile_enabled);
	err = _stable_map_gk_buffers(cli, num_points);
	bench_end(cli->profiling.buffer_read, cli->profile_enabled);

	if (err) {
		stablecl_log(log_err, "Error reading results from the GPU: %s (%d)", opencl_strerr(err), err);
		return err;
	}

	if (cli->profile_enabled)
		stable_retrieve_profileinfo(cli, *event);

	bench_begin(cli->profiling.set_results, cli->profile_enabled);

	for (size_t i = 0; i < num_points; i++) {
		if (results_1)
			results_1[i] = cli->h_kronrod[i];

		if (results_2 && cli->copy_gauss_array)
			results_2[i] = cli->h_gauss[i];

#if STABLE_MIN_LOG <= 0
		char msg[500];
		snprintf(msg, 500, "Results set P%zu: kronrod = %.3g", i, cli->h_kronrod[i]);

		if (!cli->mode_pointgenerator)
			snprintf(msg + strlen(msg), 500 - strlen(msg), ", gauss = %.3g", cli->h_gauss[i]);

#endif

		if (errs) {
			if (cli->error_mode == error_from_results) {
				if (cli->h_kronrod[i] != 0)
					errs[i] = fabs(cli->h_kronrod[i] - cli->h_gauss[i]) / cli->h_kronrod[i];
				else
					errs[i] = fabs(cli->h_kronrod[i] - cli->h_gauss[i]);
			} else if (cli->error_mode == error_is_gauss_array)
				errs[i] = cli->h_gauss[i];

#if STABLE_MIN_LOG <= 0
			snprintf(msg + strlen(msg), 500 - strlen(msg), ", relerr = %.3g", errs[i]);
#endif
		}

#if STABLE_MIN_LOG <= 0
		// stablecl_log(log_message, msg);
#endif
	}

	bench_end(cli->profiling.set_results, cli->profile_enabled);

	cl_int retval = _stable_unmap_gk_buffers(cli);

	if (retval)
		stablecl_log(log_warning, "Error unmapping buffers: %s (%d)", opencl_strerr(retval), retval);

	return err;
}

short stable_clinteg_points_parallel(struct stable_clinteg *cli, double *x, double *results_1, double* results_2, double *errs, size_t num_points, struct StableDistStruct *dist, size_t queues)
{
	cl_int err = 0;
	size_t i;
	size_t points_per_queue, processed_points, required_queues, points_for_current_queue;

	// Ensure we have enough queues
	if (cli->env.queue_count < queues)
		opencl_set_queues(&cli->env, queues);

	required_queues = num_points / MIN_POINTS_PER_QUEUE + 1;
	required_queues = required_queues > queues ? queues : required_queues;

	points_per_queue = num_points / required_queues;
	processed_points = 0;

	for (i = 0; i < required_queues; i++) {
		if (i == required_queues - 1)
			points_for_current_queue = num_points - processed_points; // Process remaining points. We don't want to leave anything because of rounding errors.
		else
			points_for_current_queue = points_per_queue;

		opencl_set_current_queue(&cli->env, i);
		err = stable_clinteg_points_async(cli, x + processed_points, points_for_current_queue, dist, NULL);

		if (err)
			goto cleanup;

		processed_points += points_for_current_queue;
	}

	if (processed_points != num_points)
		stablecl_log(log_err, "ERROR: Not enough processed points.");

	processed_points = 0;

	for (i = 0; i < required_queues; i++) {
		if (i == required_queues - 1)
			points_for_current_queue = num_points - processed_points; // Process remaining points. We don't want to leave anything because of rounding errors.
		else
			points_for_current_queue = points_per_queue;

		opencl_set_current_queue(&cli->env, i);
		err = stable_clinteg_points_end(cli, results_1 + processed_points, results_2 + processed_points, errs ? errs + processed_points : NULL, points_for_current_queue, dist, NULL);

		if (err)
			goto cleanup;

		processed_points += points_for_current_queue;
	}

cleanup:

	if (err)
		stablecl_log(log_err, "Error on parallel evaluation.");

	return err;
}

void stable_clinteg_teardown(struct stable_clinteg *cli)
{
#ifndef SIMULATOR_BUILD

	if (!cli->mode_pointgenerator && cli->h_gauss)
		clEnqueueUnmapMemObject(opencl_get_queue(&cli->env), cli->gauss, cli->h_gauss, 0, NULL, NULL);

	if (cli->h_kronrod)
		clEnqueueUnmapMemObject(opencl_get_queue(&cli->env), cli->kronrod, cli->h_kronrod, 0, NULL, NULL);

	clEnqueueUnmapMemObject(opencl_get_queue(&cli->env), cli->args, cli->h_args, 0, NULL, NULL);
#endif

	if (!cli->mode_pointgenerator) clReleaseMemObject(cli->gauss);

	clReleaseMemObject(cli->kronrod);
	clReleaseMemObject(cli->args);

	opencl_teardown(&cli->env);
}

void stable_clinteg_printinfo()
{
	printf("Libstable - OpenCL parallel integration details:\n");
	printf(" %d points Gauss - Kronrod quadrature.\n", GK_POINTS);
	printf(" %d subdivisions, %d points per thread.\n", GK_SUBDIVISIONS, POINTS_EVAL);
	printf(" Precision used: %s.\n\n", cl_precision_type);
}

void stable_clinteg_set_mode(struct stable_clinteg* cli, clinteg_mode mode)
{
	if (mode == mode_pdf)
		cli->mode_bits = MODEMARKER_PDF;
	else if (mode == mode_cdf)
		cli->mode_bits = MODEMARKER_CDF;
	else if (mode == mode_pcdf || mode == mode_quantile) // pcdf or quantile
		cli->mode_bits = MODEMARKER_PCDF;
	else
		cli->mode_bits = MODEMARKER_RNG;

	if (mode == mode_quantile) {
		cli->kern_index = KERNIDX_QUANTILE;
		cli->copy_gauss_array = 0;
		cli->mode_pointgenerator = 0;
		cli->error_mode = error_is_gauss_array;
	} else if (mode == mode_rng) {
		cli->kern_index = KERNIDX_RNG;
		cli->copy_gauss_array = 0;
		cli->mode_pointgenerator = 1;
		cli->error_mode = error_none;
	} else { // PDF, CDF or both
		cli->kern_index = KERNIDX_INTEGRATE;
		cli->copy_gauss_array = mode == mode_pcdf;
		cli->error_mode = error_from_results;
		cli->mode_pointgenerator = 0;
	}
}

const char* stable_mode_str(clinteg_mode mode)
{
	switch (mode) {
		case mode_cdf:
			return "CDF";

		case mode_pcdf:
			return "PDF & CDF";

		case mode_pdf:
			return "PDF";

		case mode_quantile:
			return "INV";

		case mode_rng:
			return "RNG";
	}
}
