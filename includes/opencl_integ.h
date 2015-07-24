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

#ifndef OPENCL_INTEG_H
#define OPENCL_INTEG_H

#include "openclenv.h"

#include "opencl_common.h"
#include "stable_api.h"

struct StableDistStruct;

typedef enum {
	error_from_results, error_is_gauss_array
} error_mode;

struct stable_clinteg {
	double interv_begin;
	double interv_end;
	unsigned int subdivisions;
	int points_rule; // Points for GK rule.
	struct openclenv env;

	cl_precision* h_gauss;
	cl_precision* h_kronrod;
	struct stable_info* h_args;
	cl_precision* subinterval_errors;
	cl_mem gauss;
	cl_mem kronrod;
	cl_mem args;
	cl_mem points;
	double result;
	double abs_error;

	struct opencl_profile profiling;
	short profile_enabled;

	size_t kern_index;
	unsigned short mode_bits;
	short copy_gauss_array;
	error_mode error_mode;
};

typedef enum {
	mode_pdf,
	mode_cdf,
	mode_pcdf,
	mode_quantile
} clinteg_mode;

int stable_clinteg_init(struct stable_clinteg* cli, size_t platform_index);

short stable_clinteg_points(struct stable_clinteg *cli,
	double *x, double *pdf_results, double* cdf_results,
	double *errs, size_t num_points, struct StableDistStruct *dist);

void stable_clinteg_teardown(struct stable_clinteg* cli);

short stable_clinteg_points_end(struct stable_clinteg *cli,
	double *pdf_results, double* cdf_results, double* errs,
	size_t num_points, struct StableDistStruct *dist,
	cl_event* event);

short stable_clinteg_points_async(struct stable_clinteg *cli,
	double *x, size_t num_points, struct StableDistStruct *dist,
	cl_event* event);

short stable_clinteg_points_parallel(struct stable_clinteg *cli,
	double *x, double *pdf_results, double* cdf_results,
	double *errs, size_t num_points,
	struct StableDistStruct *dist, size_t queues);

void stable_clinteg_printinfo();

void stable_clinteg_set_mode(struct stable_clinteg* cli, clinteg_mode mode);

const char* stable_mode_str(clinteg_mode mode);

#endif

