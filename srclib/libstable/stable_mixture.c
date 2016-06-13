#include "stable_api.h"
#include "gamma.h"

#include <signal.h>
#include <unistd.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics_double.h>

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
	size_t initial_components;
	size_t i, j, k, comp_idx, param_idx, fitter_idx;
	double dist_params[MAX_STABLE_PARAMS], new_params[MAX_STABLE_PARAMS];
	size_t num_fitter_dists = dist->max_mixture_components * MAX_STABLE_PARAMS * NUM_ALTERNATIVES_PARAMETER;
	StableDist* component;
	double previous_pdf[length];
	double previous_param_probs[num_fitter_dists];
	double changed_parameters[num_fitter_dists];
	double pdf[length];
	double jump_probability;
	size_t streak_without_change = 0;
	size_t num_changes = 0;
	double prior_mu_mean, prior_mu_variance;
	double prior_sigma_alpha, prior_sigma_beta, prior_sigma_mean, prior_sigma_variance;
	double dirichlet_initial = 1; // Maaaaaagic.
	double dirichlet_params[dist->max_mixture_components];
	double param_probability;
	double data_mean, data_variance;
	double previous_weights[dist->max_mixture_components];

	FILE* debug_data = fopen("mixture_debug.dat", "w");

	// Only set a random number of components initially when the jump on the number is done
	// via MonteCarlo reversible jumps.
	if (dist->num_mixture_components == 0) {
		initial_components = gsl_rng_uniform_int(dist->gslrand, dist->max_mixture_components) + 1;
		stable_set_mixture_components(dist, initial_components);
	}

	// Prepare the arguments for the priors.
	// TODO: Do something better than this for the estimation.
	data_mean = gsl_stats_mean(data, 1, length);
	data_variance = gsl_stats_variance(data, 1, length);

	prior_mu_mean = 0;
	prior_mu_variance = 5;

	// TODO: Wild guess. Probably really wrong.
	prior_sigma_mean = 0.5;
	prior_sigma_variance = 4;

	// Solve for α, β in the equations for mean and variance of the inverse gamma.
	prior_sigma_alpha = prior_sigma_mean / prior_sigma_variance + 2;
	prior_sigma_beta = (prior_sigma_alpha - 1) * prior_sigma_mean;

	printf("Initial parameters: %lf/%lf | %lf %lf %lf %lf\n", data_mean, data_variance, prior_mu_mean, prior_mu_variance, prior_sigma_mean, prior_sigma_variance);

	stable_activate_gpu(dist);

	for (i = 0; i < dist->num_mixture_components; i++) {
		// TODO: More magic.
		// dist->mixture_weights[i] = ((double) 1) / dist->num_mixture_components;

		// Prepare a random distribution for the parameters of each component
		stable_getparams_array(dist->mixture_components[i], new_params);

		for (param_idx = 0; param_idx < MAX_STABLE_PARAMS; param_idx++)
			new_params[param_idx] = rand_generators[param_idx](dist->mixture_components[i]);

		stable_setparams_array(dist->mixture_components[i], new_params);
		dist->mixture_components[i]->mixture_montecarlo_variance = 5;
		previous_param_probs[i] = 1;
	}

	signal(SIGINT, handle_signal);
	signal(SIGTERM, handle_signal);

	for (i = 0; i < MAX_MIXTURE_ITERATIONS && !stop; i++) {
		// Async launch of all the integration orders.
		for (param_idx = 0; param_idx < MAX_STABLE_PARAMS; param_idx++) {
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
					stable_pdf_gpu(dist, data, length, pdf, NULL);

					jump_probability = 1;

					for (k = 0; k < length; k++)
						jump_probability *= pdf[k] / previous_pdf[k];

					/* if (param_idx == STABLE_PARAM_MU)
						param_probability = gsl_ran_gaussian_pdf(new_params[param_idx] - prior_mu_mean, sqrt(prior_mu_variance));
					else if (param_idx == STABLE_PARAM_SIGMA)
						param_probability = invgamma_pdf(prior_sigma_alpha, prior_sigma_beta, new_params[param_idx]);
					else */
					param_probability = 1;

					jump_probability *= (param_probability) / (previous_param_probs[param_idx]);
					// jump_probability = exp(jump_probability);

					printf("Jump %lf\n", jump_probability);

					// Only try to do the jump if we have previous likelihoods
					if (i > 0) {
						if (rand_event(dist->gslrand, jump_probability)) {
							num_changes++;
							memcpy(previous_pdf, pdf, sizeof(double) * length);
							previous_param_probs[param_idx] = param_probability;
						} else   // Change not accepted, revert to the previous value
							stable_setparams_array(component, dist_params);
					} else
						memcpy(previous_pdf, pdf, sizeof(double) * length);

				}
			}
		}

		// Estimate the weights. TODO: Not completely sure of this.
		if (0) {
			memcpy(previous_weights, dist->mixture_weights, dist->max_mixture_components * sizeof(double));

			for (comp_idx = 0; comp_idx < dist->num_mixture_components; comp_idx++)
				dirichlet_params[comp_idx] = dirichlet_initial + length * dist->mixture_weights[comp_idx];

			gsl_ran_dirichlet(dist->gslrand, dist->num_mixture_components, dirichlet_params, dist->mixture_weights);

			stable_pdf_gpu(dist, data, length, pdf, NULL);

			jump_probability = 1;

			for (k = 0; k < length; k++)
				jump_probability *= pdf[k] / previous_pdf[k];

			if (rand_event(dist->gslrand, jump_probability)) {
				num_changes++;
				memcpy(previous_pdf, pdf, sizeof(double) * length);
			} else   // Change not accepted, revert to the previous value
				memcpy(dist->mixture_weights, previous_weights, dist->max_mixture_components * sizeof(double));
		}

		if (num_changes == 0)
			streak_without_change++;
		else
			streak_without_change = 0;

		fprintf(debug_data, "%zu", num_changes);

		for (j = 0; j < dist->num_mixture_components; j++) {
			fprintf(debug_data, " %lf %lf %lf %lf %lf",
					dist->mixture_components[j]->alfa, dist->mixture_components[j]->beta,
					dist->mixture_components[j]->mu_0, dist->mixture_components[j]->sigma,
					dist->mixture_weights[j]);
		}

		fprintf(debug_data, "\n");
		fflush(debug_data);

		num_changes = 0;

		for (comp_idx = 0; comp_idx < dist->num_mixture_components; comp_idx++) {
			component = dist->mixture_components[comp_idx];
			// component->mixture_montecarlo_variance *= 0.99999;
		}
	}

#ifdef DEBUG
	fclose(debug_data);
#endif

	return 0;
}
