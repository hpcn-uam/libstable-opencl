#include "stable_gridfit.h"

#define DIM_ALPHA 0
#define DIM_BETA 1
#define DIM_MU 2
#define DIM_SIGMA 3

static double initial_point_separation[] = { 0.2, 0.2, 0.2, 0.2 };
static double initial_contracting_coefs[] = { 0.7, 0.8, 0.6, 0.8 };

static void get_params_from_dist(StableDist* dist, double params[4])
{
	params[DIM_ALPHA] = dist->alfa;
	params[DIM_BETA] = dist->beta;
	params[DIM_MU] = dist->mu_0;
	params[DIM_SIGMA] = dist->sigma;
}

static short set_params_to_dist(StableDist* dist, double* params, size_t params_count)
{
	double dist_current_params[MAX_STABLE_PARAMS];

	get_params_from_dist(dist, dist_current_params);

	for(size_t dim = 0; dim < params_count; dim++)
		dist_current_params[dim] = params[dim];


	return stable_setparams(dist, dist_current_params[DIM_ALPHA], dist_current_params[DIM_BETA],
		dist_current_params[DIM_MU], dist_current_params[DIM_SIGMA], 0) == NOVALID;
}

static void calculate_upperleft_corner_point(struct stable_gridfit* gridfit)
{
	double dst_to_border;

	for(size_t dim = 0; dim < gridfit->fitter_dimensions; dim++)
	{
		dst_to_border = gridfit->point_sep[dim] * ((double) gridfit->fitter_per_dimension[dim] - 1) / 2;
		gridfit->corners[dim] = gridfit->centers[dim] - dst_to_border;
	}
}

static short prepare_grid_params_for_fitter(struct stable_gridfit* gridfit, size_t fitter)
{
	size_t grid_coordinate;
	double params[gridfit->fitter_dimensions];
	size_t previous_dim_acc_size = 1;

	previous_dim_acc_size = 1;

	for(size_t dim = 0; dim < gridfit->fitter_dimensions; dim++)
	{
		grid_coordinate = (fitter / previous_dim_acc_size) % gridfit->fitter_per_dimension[dim];
		previous_dim_acc_size *= gridfit->fitter_per_dimension[dim];
		params[dim] = gridfit->corners[dim] + grid_coordinate * gridfit->point_sep[dim];
	}

	return set_params_to_dist(gridfit->fitter_dists[fitter], params, gridfit->fitter_dimensions);
}

static void point_sep_iterate(struct stable_gridfit* gridfit)
{
	for(size_t dim = 0; dim < gridfit->fitter_dimensions; dim++)
		gridfit->point_sep[dim] *= gridfit->contracting_coefs[dim];
}

static void gridfit_init(struct stable_gridfit* gridfit, StableDist *dist, const double *data, const unsigned int length)
{
	gridfit->data = data;
	gridfit->data_length = length;
	gridfit->fitter_dimensions = MAX_STABLE_PARAMS;
	gridfit->fitter_dist_count = 1;
	gridfit->current_iteration = 0;
	gridfit->parallel = dist->parallel_gridfit;

	for(size_t i = 0; i < gridfit->fitter_dimensions; i++)
	{
		gridfit->fitter_per_dimension[i] = GRIDFIT_TEST_PER_DIM; // Same size for every dimension, for now;
		gridfit->fitter_dist_count *= gridfit->fitter_per_dimension[i];
	}

	gridfit->initial_dist = dist;
	gridfit->fitter_dists = calloc(gridfit->fitter_dist_count, sizeof(StableDist*));
	gridfit->waiting_events = calloc(gridfit->fitter_dist_count, sizeof(cl_event));
	gridfit->likelihoods = calloc(gridfit->fitter_dist_count, sizeof(double));

	for(size_t i = 0; i < gridfit->fitter_dist_count; i++)
		gridfit->fitter_dists[i] = stable_create(1, 0.5, 1, 1, 0);

	stable_activate_gpu(dist);
	gridfit->cli = &dist->cli;
	opencl_set_queues(&gridfit->cli->env, gridfit->fitter_dist_count);

	memcpy(gridfit->point_sep, initial_point_separation, gridfit->fitter_dimensions * sizeof(double));
	memcpy(gridfit->contracting_coefs, initial_contracting_coefs, gridfit->fitter_dimensions * sizeof(double));
}

static void set_new_center(struct stable_gridfit* gridfit, double* params)
{
	memcpy(gridfit->centers, params, gridfit->fitter_dimensions);
}

static void gridfit_iterate(struct stable_gridfit* gridfit)
{
	double pdf[gridfit->data_length];
	StableDist* dist;

	gridfit->max_likelihood = DBL_MIN;
	gridfit->min_likelihood = DBL_MAX;

	for(size_t i = 0; i < gridfit->fitter_dist_count; i++)
	{
		dist = gridfit->fitter_dists[i];

		if(prepare_grid_params_for_fitter(gridfit, i) == 0)
			stable_clinteg_points(gridfit->cli, (double*) gridfit->data, pdf, NULL, gridfit->data_length, dist);
		else
			continue;

		gridfit->likelihoods[i] = 0;

		for(size_t point = 0; point < gridfit->data_length; point++)
			gridfit->likelihoods[i] += -log(pdf[point]);

		if(gridfit->likelihoods[i] > gridfit->max_likelihood)
			gridfit->max_likelihood = gridfit->likelihoods[i];

		if(gridfit->likelihoods[i] < gridfit->min_likelihood)
		{
			gridfit->min_likelihood = gridfit->likelihoods[i];
			gridfit->min_fitter = i;
		}
	}
}

static void gridfit_iterate_parallel(struct stable_gridfit* gridfit)
{
	double pdf[gridfit->data_length];
	short fitter_enabled[gridfit->fitter_dist_count];
	StableDist* dist;

	gridfit->max_likelihood = DBL_MIN;
	gridfit->min_likelihood = DBL_MAX;
	bzero(fitter_enabled, gridfit->fitter_dist_count * sizeof(short));

	for(size_t i = 0; i < gridfit->fitter_dist_count; i++)
	{
		dist = gridfit->fitter_dists[i];

		if(prepare_grid_params_for_fitter(gridfit, i) == 0)
		{
			opencl_set_current_queue(&gridfit->cli->env, i);
			stable_clinteg_points_async(gridfit->cli, (double*) gridfit->data, gridfit->data_length, dist, NULL);
			fitter_enabled[i] = 1;
		}
	}

	for(size_t i = 0; i < gridfit->fitter_dist_count; i++)
	{
		if(!fitter_enabled[i])
			continue;

		opencl_set_current_queue(&gridfit->cli->env, i);
		stable_clinteg_points_end(gridfit->cli, pdf, NULL, gridfit->data_length, dist, NULL);

		gridfit->likelihoods[i] = 0;

		for(size_t point = 0; point < gridfit->data_length; point++)
			gridfit->likelihoods[i] += -log(pdf[point]);

		if(gridfit->likelihoods[i] > gridfit->max_likelihood)
			gridfit->max_likelihood = gridfit->likelihoods[i];

		if(gridfit->likelihoods[i] < gridfit->min_likelihood)
		{
			gridfit->min_likelihood = gridfit->likelihoods[i];
			gridfit->min_fitter = i;
		}
	}
}

int stable_fit_grid(StableDist *dist, const double *data, const unsigned int length)
{
	struct stable_gridfit gridfit;
	double likelihood_diff = DBL_MAX;
	double best_params[MAX_STABLE_PARAMS];

	gridfit_init(&gridfit, dist, data, length);
	get_params_from_dist(dist, gridfit.centers);

	while(gridfit.current_iteration < MAX_ITERATIONS && likelihood_diff > WANTED_PRECISION)
	{
		calculate_upperleft_corner_point(&gridfit);

		if(gridfit.parallel)
			gridfit_iterate_parallel(&gridfit);
		else
			gridfit_iterate(&gridfit);

		get_params_from_dist(gridfit.fitter_dists[gridfit.min_fitter], best_params);
		set_new_center(&gridfit, best_params);

		point_sep_iterate(&gridfit);

		likelihood_diff = gridfit.max_likelihood - gridfit.min_likelihood;
		gridfit.current_iteration++;
	}

	set_params_to_dist(dist, best_params, gridfit.fitter_dimensions);

	return 0;
}
