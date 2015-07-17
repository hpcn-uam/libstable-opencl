#ifndef OPENCL_INTEG_H
#define OPENCL_INTEG_H

#include "openclenv.h"

#include "opencl_common.h"
#include "stable_api.h"

struct StableDistStruct;

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
};

typedef enum {
	clinteg_pdf,
	clinteg_cdf
} clinteg_type;


int stable_clinteg_init(struct stable_clinteg* cli, size_t platform_index);

short stable_clinteg_points(struct stable_clinteg *cli,
	double *x, double *pdf_results, double *errs, size_t num_points,
	struct StableDistStruct *dist, clinteg_type type);

void stable_clinteg_teardown(struct stable_clinteg* cli);

short stable_clinteg_points_end(struct stable_clinteg *cli,
	double *pdf_results, double* errs, size_t num_points,
	struct StableDistStruct *dist, cl_event* event);

short stable_clinteg_points_async(struct stable_clinteg *cli,
	double *x, size_t num_points, struct StableDistStruct *dist,
	cl_event* event, clinteg_type type);

short stable_clinteg_points_parallel(struct stable_clinteg *cli,
	double *x, double *pdf_results, double *errs, size_t num_points,
	struct StableDistStruct *dist, size_t queues, clinteg_type type);

void stable_clinteg_printinfo();

#endif

