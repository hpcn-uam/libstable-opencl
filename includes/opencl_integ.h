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
	cl_precision* subinterval_errors;
	double result;
	double abs_error;
};


int stable_clinteg_init(struct stable_clinteg* cli);
double stable_clinteg_integrate(struct stable_clinteg* cli, double a, double b, 
		double epsabs, double epsrel, unsigned short limit,
    	double *result, double *abserr, struct StableDistStruct* dist);
void stable_clinteg_teardown(struct stable_clinteg* cli);


#endif

