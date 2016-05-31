#include "stable_api.h"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#define MAX_MIXTURE_ITERATIONS 1000
#define NUM_ALTERNATIVES_PARAMETER 1 // Number of alternative parameter values considered.

/**
 * A common function for evaluation of mixtures, calling a base function.
 */
void _stable_evaluate_mixture(StableDist *dist, const double x[], const int Nx,
                              double *result1, double *result2,
                              array_evaluator eval)
{
	size_t i, j;
	double* res1_component = NULL;
	double* res2_component = NULL;

	if (dist->is_mixture) {
		if (result1) {
			memset(result1, 0, sizeof(double) * Nx);
			res1_component = malloc(sizeof(double) * Nx);
		}

		if (result2) {
			memset(result2, 0, sizeof(double) * Nx);
			res2_component = malloc(sizeof(double) * Nx);
		}

		for (i = 0; i < dist->num_mixture_components; i++) {
			eval(dist->mixture_components[i], x, Nx, res1_component, res2_component);

			for (j = 0; j < Nx; j++) {
				if (result1) result1[j] += res1_component[j] * dist->mixture_weights[i];
				if (result2) result2[j] += res2_component[j] * dist->mixture_weights[i];
			}
		}
	}
}

static double _draw_rand(StableDist* dist, double mn, double mx, double mean) {
	double rnd = gsl_ran_gaussian(dist->gslrand, dist->mixture_montecarlo_variance) + mean;

	return max(mx, min(mn, rnd));
}

static double _draw_rand_alpha(StableDist *dist) { return _draw_rand(dist, 0.05, 1.95, dist->alfa); }
static double _draw_rand_beta(StableDist *dist) { return _draw_rand(dist, -0.95, 0.95, dist->beta); }
static double _draw_rand_mu(StableDist *dist) { return _draw_rand(dist, DBL_MIN, DBL_MAX, dist->mu_0); }
static double _draw_rand_sigma(StableDist *dist) { return _draw_rand(dist, DBL_MIN, DBL_MAX, dist->sigma); }

typedef double (*new_rand_param)(StableDist*);

static new_rand_param rand_generators[] = { _draw_rand_alpha, _draw_rand_beta, _draw_rand_mu, _draw_rand_sigma };

/**
 * Generates 1 or 0 with a certain probability.
 * @param  rnd        Random generator
 * @param  prob_event Probability of 1.
 * @return            1 or 0.
 */
static short rand_event(gsl_rng* rnd, double prob_event)
{
	return gsl_rng_uniform(rnd) <= prob_event;
}

int stable_fit_mixture(StableDist * dist, const double * data, const unsigned int length)
{
	// size_t initial_components;
	size_t i, j, k, comp_idx, param_idx, fitter_idx;
	double dist_params[MAX_STABLE_PARAMS], new_params[MAX_STABLE_PARAMS];
	size_t num_fitter_dists = dist->max_mixture_components * MAX_STABLE_PARAMS * NUM_ALTERNATIVES_PARAMETER;
	StableDist* component;
	double previous_likelihoods[num_fitter_dists];
	double likelihood;
	double changed_parameters[num_fitter_dists];
	double pdf[length];
	double jump_probability;

	// Only set a random number of components initially when the jump on the number is done
	// via MonteCarlo reversible jumps.
	// initial_components = gsl_rng_uniform_int(dist->gslrand, dist->max_mixture_components) + 1;
	// stable_set_mixture_components(dist, initial_components);

	// Configure the GPU
	stable_activate_gpu(dist);
	opencl_set_queues(&dist->cli.env, num_fitter_dists);
	stable_clinteg_set_mode(&dist->cli, mode_pdf);

	for (i = 0; i < dist->num_mixture_components; i++) {
		// Initialize the weights as random.
		dist->mixture_weights[i] = ((double) 1) / dist->num_mixture_components;

		// Prepare a random distribution for the parameters of each component
		for (param_idx = 0; param_idx < MAX_STABLE_PARAMS; param_idx++)
			new_params[param_idx] = rand_generators[param_idx](dist->mixture_components[i]);

		stable_setparams_array(dist->mixture_components[i], new_params);
	}

	for (i = 0; i < MAX_MIXTURE_ITERATIONS; i++) {
		// Async launch of all the integration orders.
		for (comp_idx = 0; comp_idx < dist->num_mixture_components; comp_idx++) {
			component = dist->mixture_components[comp_idx];

			stable_getparams_array(component, dist_params);

			for (param_idx = 0; param_idx < MAX_STABLE_PARAMS; param_idx++) {
				for (j = 0; j < NUM_ALTERNATIVES_PARAMETER; j++) {
					fitter_idx = comp_idx * MAX_STABLE_PARAMS * NUM_ALTERNATIVES_PARAMETER
					             + param_idx * NUM_ALTERNATIVES_PARAMETER + j;

					memcpy(new_params, dist_params, sizeof new_params);

					// Generate a new parameter and set it in the distrubtion
					new_params[param_idx] = rand_generators[param_idx](component);
					changed_parameters[fitter_idx] = new_params[param_idx];

					// Calculate the PDF in those points, asynchronously
					stable_setparams_array(dist, new_params);
					opencl_set_current_queue(&dist->cli.env, fitter_idx);
					stable_clinteg_points_async(&dist->cli, (double*) data, length, dist, NULL);
				}
			}
		}

		// Now let's collect the results
		for (comp_idx = 0; comp_idx < dist->num_mixture_components; comp_idx++) {
			component = dist->mixture_components[comp_idx];

			stable_getparams_array(component, new_params);

			for (param_idx = 0; param_idx < MAX_STABLE_PARAMS; param_idx++) {
				for (j = 0; j < NUM_ALTERNATIVES_PARAMETER; j++) {
					fitter_idx = comp_idx * MAX_STABLE_PARAMS * NUM_ALTERNATIVES_PARAMETER
					             + param_idx * NUM_ALTERNATIVES_PARAMETER + j;

					opencl_set_current_queue(&dist->cli.env, fitter_idx);
					stable_clinteg_points_end(&dist->cli, pdf, NULL, NULL, length, dist, NULL);

					likelihood = 0;

					for (k = 0; k < length; k++)
						likelihood += pdf[k];

					// Only try to do the jump if we have previous likelihoods
					if (i > 0) {
						jump_probability = likelihood / previous_likelihoods[fitter_idx];

						if (rand_event(dist->gslrand, jump_probability))
							new_params[param_idx] = changed_parameters[fitter_idx];
					}

					previous_likelihoods[fitter_idx] = likelihood;
				}
			}

			stable_setparams_array(component, new_params);
		}
	}

	return 0;
}
