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
	double result;
	double abs_error;

	struct opencl_profile profiling;
	short profile_enabled;
};


int stable_clinteg_init(struct stable_clinteg* cli);
double stable_clinteg_integrate(struct stable_clinteg* cli, double a, double b, 
		double epsabs, double epsrel, unsigned short limit,
    	double *result, double *abserr, struct StableDistStruct* dist);
void stable_clinteg_teardown(struct stable_clinteg* cli);


#endif

