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
#ifndef STABLE_GRIDFIT_H
#define STABLE_GRIDFIT_H

#include "stable_api.h"
#include "opencl_integ.h"

#define GRIDFIT_TEST_PER_DIM 3
#define MAX_STABLE_PARAMS 4
#define ESTIMATING_PARAMS 2
#define MAX_ITERATIONS 10 // Tip: Use the iterations_calc script in scripts folder
#define WANTED_PRECISION 0.015
#define MIN_LIKELIHOOD_DIFF 0.01

struct stable_gridfit {
	StableDist* initial_dist;
	StableDist** fitter_dists;
	size_t fitter_dist_count;
	size_t fitter_dimensions;
	size_t fitter_per_dimension[MAX_STABLE_PARAMS];
	unsigned int current_iteration;
	double *data;
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
	size_t max_fitter;
	struct stable_clinteg* cli;
	short parallel;
	double mc_c;
	double mc_z;
};

int stable_fit_grid(StableDist *dist, const double *data, const unsigned int length);
void stable_gridfit_destroy(struct stable_gridfit* gridfit);

#endif
