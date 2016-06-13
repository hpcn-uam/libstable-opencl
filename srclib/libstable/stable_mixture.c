#include "stable_api.h"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#define MAX_MIXTURE_ITERATIONS 10000
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

static double _draw_rand(StableDist* dist, double mn, double mx, double mean)
{
	double rnd = gsl_ran_gaussian(dist->gslrand, dist->mixture_montecarlo_variance) + mean;

	return min(mx, max(mn, rnd));
}

static double _draw_rand_alpha(StableDist *dist)
{
	return _draw_rand(dist, 0.0, 2.0, dist->alfa);
}

static double _draw_rand_beta(StableDist *dist)
{
	return _draw_rand(dist, -1, 1, dist->beta);
}

static double _draw_rand_mu(StableDist *dist)
{
	return _draw_rand(dist, -DBL_MAX, DBL_MAX, dist->mu_0);
}

static double _draw_rand_sigma(StableDist *dist)
{
	return _draw_rand(dist, 0, DBL_MAX, dist->sigma);
}

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

volatile sig_atomic_t stop = 0;

void handle_signal(int sig)
{
	stop = 1;
}

int stable_fit_mixture(StableDist * dist, const double * data, const unsigned int length)
{
	// size_t initial_components;
	size_t i, j, k, comp_idx, param_idx, fitter_idx;
	double dist_params[MAX_STABLE_PARAMS], new_params[MAX_STABLE_PARAMS];
	size_t num_fitter_dists = dist->max_mixture_components * MAX_STABLE_PARAMS * NUM_ALTERNATIVES_PARAMETER;
	StableDist* component;
	double previous_pdf[num_fitter_dists][length];
	double likelihood;
	double changed_parameters[num_fitter_dists];
	double pdf[length];
	double jump_probability;
	size_t streak_without_change = 0;
	size_t num_changes = 0;

#ifdef DEBUG
	FILE* debug_data = fopen("mixture_debug.dat", "w");
#endif

	// Only set a random number of components initially when the jump on the number is done
	// via MonteCarlo reversible jumps.
	// initial_components = gsl_rng_uniform_int(dist->gslrand, dist->max_mixture_components) + 1;
	// stable_set_mixture_components(dist, initial_components);



	// stable_activate_gpu(dist);
	stable_set_THREADS(1);

	for (i = 0; i < dist->num_mixture_components; i++) {
		// Initialize the weights as random.
		// dist->mixture_weights[i] = ((double) 1) / dist->num_mixture_components;

		// Prepare a random distribution for the parameters of each component
		stable_getparams_array(dist->mixture_components[i], new_params);

		for (param_idx = 0; param_idx < 2; param_idx++)
			new_params[param_idx] = rand_generators[param_idx](dist->mixture_components[i]);

		stable_setparams_array(dist->mixture_components[i], new_params);
		dist->mixture_components[i]->mixture_montecarlo_variance = 2;
	}

	signal(SIGINT, handle_signal);
	signal(SIGTERM, handle_signal);

	for (i = 0; i < MAX_MIXTURE_ITERATIONS && !stop; i++) {
		// Async launch of all the integration orders.
		for (param_idx = 0; param_idx < 2; param_idx++) {
			for (j = 0; j < NUM_ALTERNATIVES_PARAMETER; j++) {
				for (comp_idx = 0; comp_idx < dist->num_mixture_components; comp_idx++) {
					component = dist->mixture_components[comp_idx];

					stable_getparams_array(component, dist_params);

					fitter_idx = param_idx * dist->num_mixture_components * NUM_ALTERNATIVES_PARAMETER
								 + j * dist->num_mixture_components + comp_idx;

					memcpy(new_params, dist_params, sizeof new_params);

					// Generate a new parameter and set it in the distrubtion
					new_params[param_idx] = rand_generators[param_idx](component);
					changed_parameters[fitter_idx] = new_params[param_idx];

					stable_setparams_array(component, new_params);

#ifdef DEBUG
					fprintf(stderr, "Iter %zu: Fitter %zu testing component %zu, param %zu, alt %zu\n",
							i, fitter_idx, comp_idx, param_idx, j);
#endif
					stable_pdf(dist, data, length, pdf, NULL);

					jump_probability = 1;

					for (k = 0; k < length; k++)
						jump_probability *= pdf[k] / previous_pdf[fitter_idx][k];

					printf("Jump %lf\n", jump_probability);

					// Only try to do the jump if we have previous likelihoods
					if (i > 0) {
						if (rand_event(dist->gslrand, jump_probability)) {
							num_changes++;
							memcpy(previous_pdf[fitter_idx], pdf, sizeof(double) * length);
						} else   // Change not accepted, revert to the previous value
							stable_setparams_array(component, dist_params);
					}

				}
			}
		}

		if (num_changes == 0)
			streak_without_change++;
		else
			streak_without_change = 0;

#ifdef DEBUG
		fprintf(debug_data, "%zu", num_changes);

		for (j = 0; j < dist->num_mixture_components; j++) {
			fprintf(debug_data, " %lf %lf %lf %lf", num_changes,
					dist->mixture_components[j]->alfa, dist->mixture_components[j]->beta,
					dist->mixture_components[j]->mu_0, dist->mixture_components[j]->sigma);
		}

		fprintf(debug_data, "\n");
		fflush(debug_data);
#endif

		num_changes = 0;

		for (comp_idx = 0; comp_idx < dist->num_mixture_components; comp_idx++) {
			component = dist->mixture_components[comp_idx];
			component->mixture_montecarlo_variance *= 0.9999;
		}
	}

#ifdef DEBUG
	fclose(debug_data);
#endif

	return 0;
}
