#ifndef STABLE_GRIDFIT_H
#define STABLE_GRIDFIT_H

#include "stable_api.h"
#include "opencl_integ.h"

#define GRIDFIT_TEST_PER_DIM 2
#define MAX_STABLE_PARAMS 4
#define MAX_ITERATIONS 200 // Tip: Use the iterations_calc script in scripts folder
#define WANTED_PRECISION 1

struct stable_gridfit {
	StableDist* initial_dist;
	StableDist** fitter_dists;
	size_t fitter_dist_count;
	size_t fitter_dimensions;
	size_t fitter_per_dimension[MAX_STABLE_PARAMS];
	unsigned int current_iteration;
	const double *data;
	size_t data_length;
	double corners[MAX_STABLE_PARAMS];
	double centers[MAX_STABLE_PARAMS];
	double point_sep[MAX_STABLE_PARAMS];
	double contracting_coefs[MAX_STABLE_PARAMS];
	cl_event* waiting_events;
	double* likelihoods;
	double max_likelihood;
	double min_likelihood;
	size_t min_fitter;
	struct stable_clinteg* cli;
	short parallel;
};

int stable_fit_grid(StableDist *dist, const double *data, const unsigned int length);

#endif
