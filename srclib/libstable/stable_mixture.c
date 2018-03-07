#include "stable_api.h"
#include "kde.h"
#include "stable_gridfit.h"
#include "methods.h"
#include "stable_mixture_initialestim.h"

#include <signal.h>
#include <unistd.h>
#include <time.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_gamma.h>

#define RNG_STD 0.05

#define VERBOSE_MIXTURE
#define VERBOSE_SPLITCOMBINE
#define DEBUG

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

		if (res1_component) free(res1_component);

		if (res2_component) free(res2_component);
	}
}

static double _draw_rand(StableDist* dist, double mn, double mx, double mean, double std)
{
	double rnd = gsl_ran_gaussian(dist->gslrand, std) + mean;

	return min(mx, max(mn, rnd));
}

static double _draw_rand_alpha(StableDist *dist)
{
	return _draw_rand(dist, 0.001, 2.0, dist->alfa, RNG_STD);
}

static double _draw_rand_beta(StableDist *dist)
{
	return _draw_rand(dist, -1, 1, dist->beta, RNG_STD);
}

static double _draw_rand_mu(StableDist *dist)
{
	return _draw_rand(dist, -DBL_MAX, DBL_MAX, dist->mu_0, RNG_STD);
}

static double _draw_rand_sigma(StableDist *dist)
{
	return _draw_rand(dist, 0.001, DBL_MAX, dist->sigma, RNG_STD);
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
	printf("Stopping...\n");
	stop = 1;
}

static void _do_component_split(
	StableDist * dist, size_t comp_to_split, double w1, double w2,
	double params_1[4], double params_2[4], size_t new_comp_index)
{
	size_t split_1 = comp_to_split;
	size_t split_2 = dist->num_mixture_components; // Component 2 is created at the last position

	stable_set_mixture_components(dist, dist->num_mixture_components + 1);

	dist->mixture_weights[split_1] = w1;
	dist->mixture_weights[split_2] = w2;

	stable_setparams_array(dist->mixture_components[split_1], params_1);
	stable_setparams_array(dist->mixture_components[split_2], params_2);

	if (new_comp_index != split_2)
		stable_swap_components(dist, split_2, new_comp_index);

#ifdef VERBOSE_SPLITCOMBINE
	printf("Set split %zu into %zu and %zu (swapping last with %zu)\n", comp_to_split, split_1, split_2, new_comp_index);
#endif
}

static size_t _do_component_combine(
	StableDist * dist, size_t comp_1, size_t comp_2, double w_comb,
	double params[4], size_t* removed_index)
{
	size_t combined_comp = min(comp_1, comp_2);
	size_t removed_comp = max(comp_1, comp_2);

	if (removed_index != NULL)
		*removed_index = removed_comp;

	// Not the last component, replace this one with the last
	if (removed_comp < dist->num_mixture_components - 1)
		stable_swap_components(dist, dist->num_mixture_components - 1, removed_comp);

	stable_set_mixture_components(dist, dist->num_mixture_components - 1);
	dist->mixture_weights[combined_comp] = w_comb;
	stable_setparams_array(dist->mixture_components[combined_comp], params);

#ifdef VERBOSE_SPLITCOMBINE
	printf("Set combine %lf (%zu %zu -> %zu, rem %zu)\n", w_comb, comp_1, comp_2, combined_comp, removed_comp);
#endif

	return combined_comp;
}

size_t _iteration;

static short _calc_splitcombine_acceptance_ratio(
	StableDist * dist, const double * data, const unsigned int length, short is_split, double * current_pdf,
	size_t comp_1, size_t comp_2,
	double w1, double w2, double w_comb, double u1, double u2, double u3,
	double params_1[4], double params_2[4], double params_comb[4])
{
	double new_pdf[length];
	static FILE* fsplit = NULL;
	size_t removed_index;

	if (!fsplit)
		fsplit = fopen("mixture_split.dat", "w");

#ifdef VERBOSE_SPLITCOMBINE
	printf("Before check, μ values = ");

	for (size_t i = 0; i < dist->num_mixture_components; i++)
		printf("%lf ", dist->mixture_components[i]->mu_0);

	printf("\n");
#endif

	if (is_split) {
		comp_2 = dist->num_mixture_components; // New component is the last one
		_do_component_split(dist, comp_1, w1, w2, params_1, params_2, comp_2);
	} else
		comp_1 = _do_component_combine(dist, comp_1, comp_2, w_comb, params_comb, &removed_index);

	double sum = 0;

	for (size_t i = 0; i < dist->num_mixture_components; i++)
		sum += dist->mixture_weights[i];

	stable_pdf_gpu(dist, data, length, new_pdf, NULL);

	double n1 = length * w1;
	double n2 = length * w2;

	double mu1 = params_1[STABLE_PARAM_MU], mu2 = params_2[STABLE_PARAM_MU], mu_comb = params_comb[STABLE_PARAM_MU];
	double sigma1 = params_1[STABLE_PARAM_SIGMA], sigma2 = params_2[STABLE_PARAM_SIGMA], sigma_comb = params_comb[STABLE_PARAM_SIGMA];

	double log_likelihood_ratio = 0;

	for (size_t k = 0; k < length; k++)
		log_likelihood_ratio += log(new_pdf[k]) - log(current_pdf[k]);

	double alpha_beta_ratio = 0.25;

	double log_weight_ratio =
		log(w1) * (dist->prior_weights - 1 + n1)
		+ log(w2) * (dist->prior_weights - 1 + n2)
		- log(w_comb) * (dist->prior_weights - 1 + n1 + n2)
		- gsl_sf_lnbeta(dist->prior_weights, dist->num_mixture_components * dist->prior_weights)
		;

	log_weight_ratio = 0;

	double mu_ratio =
		sqrt(1 / (M_2_PI * dist->prior_mu_variance))
		* exp(-0.5 * (1 / dist->prior_mu_variance) * (
				  pow(mu1 - dist->prior_mu_avg, 2) + pow(mu2 - dist->prior_mu_avg, 2) - pow(mu_comb - dist->prior_mu_avg, 2)
			  ));

	double log_sigma_ratio =
		(log(dist->prior_sigma_beta0) * dist->prior_sigma_alpha0)
		- log(gsl_sf_gamma(dist->prior_sigma_alpha0))
		+ 2 * (-dist->prior_sigma_alpha0 - 1) * log(sigma1 * sigma2 / sigma_comb)
		- dist->prior_sigma_beta0 * (pow(sigma1, -2) + pow(sigma2, -2) - pow(sigma_comb, -2));

	double move_probability = 1 / (gsl_ran_beta_pdf(u1, 2, 2) * gsl_ran_beta_pdf(u2, 2, 2) * gsl_ran_beta_pdf(u3, 1, 1));

	double jacobian =
		(w_comb * fabs(mu1 - mu2) * pow(sigma1, 2) * pow(sigma2, 2))
		/ (u2 * (1 - pow(u2, 2)) * (1 - u3) * pow(sigma_comb, 2));

	double log_acceptance_ratio =
		log_likelihood_ratio
		+ log(alpha_beta_ratio) + log_weight_ratio +
		log(mu_ratio) + log_sigma_ratio + log(move_probability) + log(jacobian);

	if (!is_split)
		log_acceptance_ratio = - log_acceptance_ratio;

	double acceptance_ratio = min(1, exp(log_acceptance_ratio));

#ifdef VERBOSE_SPLITCOMBINE
	printf("Ratios: αβ = %lf, w = %lf, μ = %lf, σ = %lf, m = %lf, j = %lf, L = %lf, log L = %lf\n",
		   alpha_beta_ratio, (log_weight_ratio), mu_ratio, log_sigma_ratio, move_probability,
		   jacobian, exp(log_likelihood_ratio), log_likelihood_ratio);
	printf("Acceptance ratio: %lf\n", acceptance_ratio);
#endif

	fprintf(fsplit, "%zu %lf %lf %lf %lf\n", _iteration, mu1, mu2, mu_comb, acceptance_ratio);
	fflush(fsplit);

	if (rand_event(dist->gslrand, acceptance_ratio)) {
		memcpy(current_pdf, new_pdf, sizeof(double) * length);

		// In a split, try have as less change as possible.
		if (is_split) {
			if (fabs(mu2 - mu_comb) < fabs(mu1 - mu_comb))
				stable_swap_components(dist, comp_1, comp_2);
		}

		return 1; // Move accepted
	}

	// If move is not accepted, revert the previous operation

	if (is_split)
		_do_component_combine(dist, comp_1, comp_2, w_comb, params_comb, NULL);
	else
		_do_component_split(dist, comp_1, w1, w2, params_1, params_2, removed_index);

	sum = 0;

	for (size_t i = 0; i < dist->num_mixture_components; i++)
		sum += dist->mixture_weights[i];

#ifdef VERBOSE_SPLITCOMBINE
	printf("After check, μ values = ");

	for (size_t i = 0; i < dist->num_mixture_components; i++)
		printf("%lf ", dist->mixture_components[i]->mu_0);

	printf("\n");
#endif

	return 0;
}

static int _check_split_move(StableDist * dist, const double * data, const unsigned int length, double * current_pdf)
{
	size_t comp_idx;
	short accepted;

	if (dist->num_mixture_components == dist->max_mixture_components)
		return 0;

	comp_idx = gsl_rng_uniform_int(dist->gslrand, dist->num_mixture_components);

	double curr_weight = dist->mixture_weights[comp_idx];
	StableDist* comp = dist->mixture_components[comp_idx];

	double curr_params[4];

	stable_getparams_array(comp, curr_params);

	double u1 = gsl_ran_beta(dist->gslrand, 2, 2);
	double u2 = gsl_ran_beta(dist->gslrand, 2, 2);
	double u3 = gsl_ran_beta(dist->gslrand, 1, 1);

	double new_weight_1 = curr_weight * u1;
	double new_weight_2 = curr_weight * (1 - u1);

	double new_mu_1 = comp->mu_0 - u2 * comp->sigma * sqrt(new_weight_2 / new_weight_1);
	double new_mu_2 = comp->mu_0 + u2 * comp->sigma * sqrt(new_weight_1 / new_weight_2);

	if (new_mu_1 >= new_mu_2)
		return 0;

	double new_sigma_1 = sqrt(u3 * (1 - pow(u2, 2)) * pow(comp->sigma, 2) * curr_weight / new_weight_1);
	double new_sigma_2 = sqrt((1 - u3) * (1 - pow(u2, 2)) * pow(comp->sigma, 2) * curr_weight / new_weight_2);

	double params_1[4] = { curr_params[STABLE_PARAM_ALPHA], curr_params[STABLE_PARAM_BETA], new_mu_1, new_sigma_1 };
	double params_2[4] = { curr_params[STABLE_PARAM_ALPHA], curr_params[STABLE_PARAM_BETA], new_mu_2, new_sigma_2 };

#ifdef VERBOSE_SPLITCOMBINE
	stable_print_params_array(curr_params, "split base");
	stable_print_params_array(params_1, "split 1");
	stable_print_params_array(params_2, "split 2");

	printf("weights %lf | %lf (( %lf\n", new_weight_1, new_weight_2, curr_weight);
#endif

	accepted = _calc_splitcombine_acceptance_ratio(
				   dist, data, length, 1, current_pdf,
				   comp_idx, -1, new_weight_1, new_weight_2, curr_weight, u1, u2, u3,
				   params_1, params_2, curr_params);

#ifdef VERBOSE_SPLITCOMBINE
	stable_print_params(comp, "recovered");
#endif

	return accepted; // Only 1 proposal
}

/**
 * Randomly selects two adjacent components for a possible combine move.
 *
 * Note: components i and j are adjacent if no other component exists with mean
 * between those of i and j.
 * @param  dist   Distribution
 * @param  comp_1 Index of the first component to combine (the one with the lower mean)
 * @param  comp_2 Index of the second component to combine.
 * @return        0 if everything is ok, -1 if error.
 */
static int _search_components_to_combine(StableDist* dist, size_t* comp_1, size_t* comp_2)
{
	size_t i;
	size_t comp_indexes[dist->num_mixture_components];
	double min_avg, max_avg;

	if (dist->num_mixture_components < 2)
		return -1;

	for (i = 0; i < dist->num_mixture_components; i++)
		comp_indexes[i] = i;

	gsl_ran_shuffle(dist->gslrand, comp_indexes, dist->num_mixture_components, sizeof(size_t));

	min_avg = min(dist->mixture_components[comp_indexes[0]]->mu_0, dist->mixture_components[comp_indexes[1]]->mu_0);
	max_avg = max(dist->mixture_components[comp_indexes[0]]->mu_0, dist->mixture_components[comp_indexes[1]]->mu_0);

	// Iterate over all the components, find the one with the minimum average and the
	// maximum and ensure there are not components between them, all in one pass.
	for (i = 0; i < dist->num_mixture_components; i++) {
		if (dist->mixture_components[i]->mu_0 == min_avg)
			*comp_1 = i;
		else if (dist->mixture_components[i]->mu_0 == max_avg)
			*comp_2 = i;
		else if (dist->mixture_components[i]->mu_0 > min_avg && dist->mixture_components[i]->mu_0 < max_avg)
			return -1;
	}

	// Ensure that comp_1 < comp_2 to avoid swapping components when moves are rejected.
	if (*comp_1 > *comp_2) {
		size_t swap = *comp_1;
		*comp_1 = *comp_2;
		*comp_2 = swap;
	}

	return 0;
}

static int _check_combine_move(StableDist * dist, const double * data, const unsigned int length, double * current_pdf)
{
	size_t comp_1_idx, comp_2_idx;
	StableDist* comp_1, *comp_2;
	short accepted;

	if (_search_components_to_combine(dist, &comp_1_idx, &comp_2_idx) == -1)
		return 0; // Don't try the combine move in case of error.

	comp_1 = dist->mixture_components[comp_1_idx];
	comp_2 = dist->mixture_components[comp_2_idx];

	double params_1[4], params_2[4], params_comb[4];
	stable_getparams_array(comp_1, params_1);
	stable_getparams_array(comp_2, params_2);

	double w1 = dist->mixture_weights[comp_1_idx];
	double w2 = dist->mixture_weights[comp_2_idx];

	double alpha1 = comp_1->alfa;
	double beta1 = comp_1->beta;
	double mu1 = comp_1->mu_0;
	double sigma1 = comp_1->sigma;

	double alpha2 = comp_2->alfa;
	double beta2 = comp_2->beta;
	double mu2 = comp_2->mu_0;
	double sigma2 = comp_2->sigma;

	double w_comb = w1 + w2;
	double mu_comb = (w1 * mu1 + w2 * mu2) / w_comb;
	double sigma_comb =
		sqrt(
			(w1 * (pow(mu1, 2) + pow(sigma1, 2)) + w2 * (pow(mu2, 2) + pow(sigma2, 2)))
			/ w_comb - pow(mu_comb, 2)
		);

	params_comb[STABLE_PARAM_ALPHA] = (alpha1 + alpha2) / 2;
	params_comb[STABLE_PARAM_BETA] = (beta1 + beta2) / 2;
	params_comb[STABLE_PARAM_MU] = mu_comb;
	params_comb[STABLE_PARAM_SIGMA] = sigma_comb;

	double u1 = w1 / w_comb;
	double u2 = (mu2 - mu_comb) / (sigma_comb * sqrt(w1 / w2));
	double u3 = w1 * pow(sigma1, 2) / (pow(sigma_comb, 2) * (1 - pow(u2, 2)) * w_comb);

#ifdef VERBOSE_SPLITCOMBINE
	stable_print_params_array(params_1, "Comb 1");
	stable_print_params_array(params_2, "Comb 2");
	stable_print_params_array(params_comb, "Comb res");
#endif

	accepted = _calc_splitcombine_acceptance_ratio(
				   dist, data, length, 0, current_pdf,
				   comp_1_idx, comp_2_idx, w1, w2, w_comb, u1, u2, u3,
				   params_1, params_2, params_comb);

	return accepted;
}

void stable_fit_mixture_default_settings(struct stable_mcmc_settings* settings)
{
	settings->max_iterations = 10000;
	settings->burnin_period = 300;
	settings->num_alternative_parameters = 1;
	settings->fix_components = 0;
	settings->estimate_weight = 1;
	settings->skip_initial_estimation = 0;

	strncpy(settings->debug_data_fname, "mixture_debug.dat", sizeof(settings->debug_data_fname));
}

static void _mcmc_settings_allocate_arrays(struct stable_mcmc_settings* settings, size_t max_components)
{
	size_t i, j;

	settings->param_values = calloc(max_components, sizeof(double **));
	settings->correlations = calloc(max_components, sizeof(double **));
	settings->final_param_avg = calloc(max_components, sizeof(double*));
	settings->final_param_std = calloc(max_components, sizeof(double*));
	settings->final_weight_avg = calloc(max_components, sizeof(double*));
	settings->final_weight_std = calloc(max_components, sizeof(double*));
	settings->weights = calloc(max_components, sizeof(double *));

	for (i = 0; i < max_components; i++) {
		settings->param_values[i] = calloc(MAX_STABLE_PARAMS, sizeof(double*));
		settings->final_param_avg[i] = calloc(MAX_STABLE_PARAMS, sizeof(double));
		settings->final_param_std[i] = calloc(MAX_STABLE_PARAMS, sizeof(double));
		settings->weights[i] = calloc(settings->max_iterations + settings->burnin_period, sizeof(double));
		settings->correlations[i] = calloc(MAX_STABLE_PARAMS, sizeof(double**));

		for (j = 0; j < MAX_STABLE_PARAMS; j++) {
			settings->param_values[i][j] = calloc(settings->max_iterations + settings->burnin_period, sizeof(double));
			settings->correlations[i][j] = calloc(MAX_STABLE_PARAMS, sizeof(double*));
		}
	}
}

int stable_fit_mixture(StableDist * dist, const double * data, const unsigned int length)
{
	struct stable_mcmc_settings settings;

	stable_fit_mixture_default_settings(&settings);

	return stable_fit_mixture_settings(dist, data, length, &settings);
}

static void _stable_fix_mixture_finalize(StableDist *dist, const double* data, const unsigned int length, struct stable_mcmc_settings* settings);

int stable_fit_mixture_settings(StableDist *dist, const double* data, const unsigned int length, struct stable_mcmc_settings* settings)
{
	size_t i, j, k, comp_idx, param_idx, fitter_idx;
	double dist_params[MAX_STABLE_PARAMS], new_params[MAX_STABLE_PARAMS];
	size_t num_fitter_dists = dist->max_mixture_components * MAX_STABLE_PARAMS * settings->num_alternative_parameters;
	StableDist* component;
	double previous_pdf[length];
	double previous_param_probs[num_fitter_dists][MAX_STABLE_PARAMS];
	double pdf[length];
	double jump_probability;
	size_t streak_without_change = 0;
	size_t num_changes = 0;
	size_t total_changes = 0;
	size_t num_considered_moves = 0;
	double dirichlet_params[dist->max_mixture_components];
	double param_probability;
	double previous_weights[dist->max_mixture_components];
	FILE* debug_data;

	debug_data = fopen(settings->debug_data_fname, "w");

	if (!debug_data) {
		perror("fopen");
		return -1;
	}

	_mcmc_settings_allocate_arrays(settings, dist->max_mixture_components);

	gsl_set_error_handler_off();

	if (!settings->skip_initial_estimation)
		stable_mixture_prepare_initial_estimation(dist, data, length);

	stable_rnd_seed(dist, time(NULL));

	for (i = 0; i < dist->num_mixture_components; i++) {
		dist->mixture_components[i]->mixture_montecarlo_variance = 0.05;

		for (j = 0; j < MAX_STABLE_PARAMS; j++)
			previous_param_probs[i][j] = 1;
	}

	signal(SIGINT, handle_signal);
	signal(SIGTERM, handle_signal);

	fprintf(debug_data, "0 %zu", dist->num_mixture_components);

	for (j = 0; j < dist->num_mixture_components; j++) {
		fprintf(debug_data, " %lf %lf %lf %lf %lf",
				dist->mixture_components[j]->alfa, dist->mixture_components[j]->beta,
				dist->mixture_components[j]->mu_0, dist->mixture_components[j]->sigma,
				dist->mixture_weights[j]);
	}

	fprintf(debug_data, "\n");

	for (i = 0; i < settings->burnin_period + settings->max_iterations && !stop; i++) {
		// Async launch of all the integration orders.
		_iteration = i;

		for (param_idx = 0; param_idx < MAX_STABLE_PARAMS; param_idx++) {
			if (param_idx == STABLE_PARAM_MU && i < settings->location_lock_iterations)
				continue;

			for (j = 0; j < settings->num_alternative_parameters; j++) {
				for (comp_idx = 0; comp_idx < dist->num_mixture_components; comp_idx++) {
					num_considered_moves++;

					component = dist->mixture_components[comp_idx];

					stable_getparams_array(component, dist_params);

					fitter_idx = param_idx * dist->num_mixture_components * settings->num_alternative_parameters
								 + j * dist->num_mixture_components + comp_idx;

					memcpy(new_params, dist_params, sizeof new_params);

					// Generate a new parameter and set it in the distrubtion
					new_params[param_idx] = rand_generators[param_idx](component);
					stable_setparams_array(component, new_params);

#ifdef DEBUG
					fprintf(stderr, "Iter %zu: Fitter %zu testing component %zu, param %zu, alt %zu\n",
							i, fitter_idx, comp_idx, param_idx, j);
#endif
					stable_pdf_gpu(dist, data, length, pdf, NULL);

					jump_probability = 0;

					for (k = 0; k < length; k++)
						jump_probability += log(pdf[k]) - log(previous_pdf[k]);

					jump_probability = exp(jump_probability);

					/* if (param_idx == STABLE_PARAM_MU)
						param_probability = gsl_ran_gaussian_pdf(new_params[param_idx] - prior_mu_mean, sqrt(prior_mu_variance));
					else if (param_idx == STABLE_PARAM_SIGMA)
						param_probability = invgamma_pdf(prior_sigma_alpha, prior_sigma_beta, new_params[param_idx]);
					else */
					param_probability = 1;

					jump_probability *= (param_probability) / (previous_param_probs[comp_idx][param_idx]);
					// jump_probability = exp(jump_probability);

					// Only try to do the jump if we have previous likelihoods
					short accepted = 0;

					if (i > 0) {
						if (rand_event(dist->gslrand, jump_probability)) {
							accepted = 1;
							num_changes++;
							memcpy(previous_pdf, pdf, sizeof(double) * length);
							previous_param_probs[comp_idx][param_idx] = param_probability;
						} else   // Change not accepted, revert to the previous value
							stable_setparams_array(component, dist_params);
					} else
						memcpy(previous_pdf, pdf, sizeof(double) * length);

					settings->param_values[comp_idx][param_idx][i] = new_params[param_idx];

#ifdef VERBOSE_MIXTURE
					fprintf(stderr, "I%zuC%zu %s %.3lf -> %.3lf probability %.3g Accept %hd\n",
							i, comp_idx, _param_names[param_idx], dist_params[param_idx], new_params[param_idx], jump_probability, accepted);
#endif
				}
			}
		}

		// Estimate the weights. TODO: Not completely sure of this.
		if (settings->estimate_weight) {
			num_considered_moves++;

			memcpy(previous_weights, dist->mixture_weights, dist->max_mixture_components * sizeof(double));

			for (comp_idx = 0; comp_idx < dist->num_mixture_components; comp_idx++)
				dirichlet_params[comp_idx] = dist->prior_weights * dist->mixture_weights[comp_idx];

			gsl_ran_dirichlet(dist->gslrand, dist->num_mixture_components, dirichlet_params, dist->mixture_weights);

			jump_probability = 0;

			// Do not even bother with weight samples that have one component with almost zero weight.
			if (gsl_stats_min(dist->mixture_weights, 1, dist->num_mixture_components) >= 1e-3)   {
				stable_pdf_gpu(dist, data, length, pdf, NULL);

				for (k = 0; k < length; k++)
					jump_probability += log(pdf[k]) - log(previous_pdf[k]);

				jump_probability = exp(jump_probability);
			}

			short accepted = 0;

#ifdef VERBOSE_MIXTURE
			fprintf(stderr, "I%zu W ", i);

			for (j = 0; j < dist->num_mixture_components; j++)
				fprintf(stderr, "%zu(%.3lf -> %.3lf) ", j, previous_weights[j], dist->mixture_weights[j]);

#endif

			if (rand_event(dist->gslrand, jump_probability)) {
				num_changes++;
				accepted = 1;
				memcpy(previous_pdf, pdf, sizeof(double) * length);
			} else     // Change not accepted, revert to the previous value
				memcpy(dist->mixture_weights, previous_weights, dist->max_mixture_components * sizeof(double));

#ifdef VERBOSE_MIXTURE
			fprintf(stderr, " Accepted %hd\n", accepted);
#endif

			for (comp_idx = 0; comp_idx < dist->num_mixture_components; comp_idx++)
				settings->weights[comp_idx][i] = dist->mixture_weights[comp_idx];
		}

		if (!settings->fix_components) {
			if (i >= settings->location_lock_iterations && (settings->max_iterations - i) > settings->fix_components_during_last_n_iterations) {
				if (rand_event(dist->gslrand, dist->birth_probs[dist->num_mixture_components])) {
					printf("\nIteration %zu: Try split\n", i);
					num_considered_moves++;

					if (_check_split_move(dist, data, length, previous_pdf)) {
						num_changes++;
						printf("Accept split\n");
					}
				} else if (rand_event(dist->gslrand, dist->death_probs[dist->num_mixture_components])) {
					printf("\nIteration %zu: Try combine\n", i);
					num_considered_moves++;

					if (_check_combine_move(dist, data, length, previous_pdf)) {
						num_changes++;
						printf("Accept combine\n");
					}
				}
			}
		}

		//#ifdef VERBOSE_MIXTURE

		if (i == settings->location_lock_iterations)
			printf("Iteration %zu: Releasing initial lock on location and component birth/death\n", i);
		else if (settings->max_iterations - i == settings->fix_components_during_last_n_iterations)
			printf("Iteration %zu: Locking component birth/death for final refinement\n", i);

		//#endif

		if (num_changes == 0)
			streak_without_change++;
		else
			streak_without_change = 0;

#ifndef VERBOSE_MIXTURE
		// printf("\rIter %zu - %zu changes", i, num_changes);
		// fflush(stdout);
#endif
		fprintf(debug_data, "%zu %zu", num_changes, dist->num_mixture_components);

		for (j = 0; j < dist->num_mixture_components; j++) {
			fprintf(debug_data, " %lf %lf %lf %lf %lf",
					dist->mixture_components[j]->alfa, dist->mixture_components[j]->beta,
					dist->mixture_components[j]->mu_0, dist->mixture_components[j]->sigma,
					dist->mixture_weights[j]);
		}

		fprintf(debug_data, "\n");
		fflush(debug_data);

		if (i <= settings->burnin_period)
			num_considered_moves = 0;
		else
			total_changes += num_changes;

		num_changes = 0;

		if (settings->decrement_generation_variance) {
			for (comp_idx = 0; comp_idx < dist->num_mixture_components; comp_idx++) {
				component = dist->mixture_components[comp_idx];
				component->mixture_montecarlo_variance *= 0.99999;
			}
		}
	}

	settings->acceptance_ratio = ((double)total_changes) / num_considered_moves;
	settings->num_iterations = i;
	settings->num_final_components = dist->num_mixture_components;

	printf("\n");

	if (i > settings->burnin_period)
		_stable_fix_mixture_finalize(dist, data, length, settings);
	else
		printf("WARNING: Burn-in period not passed.\n");

#ifdef DEBUG
	fclose(debug_data);
#endif

	return 0;
}

static void _stable_fix_mixture_finalize(StableDist *dist, const double* data, const unsigned int length, struct stable_mcmc_settings* settings)
{
	size_t comp_idx, param_idx;
	size_t num_valid_iter = settings->num_iterations - settings->burnin_period;
	double new_params[MAX_STABLE_PARAMS];
	printf("%zu\n", num_valid_iter);

	for (comp_idx = 0; comp_idx < dist->num_mixture_components; comp_idx++) {
		for (param_idx = 0; param_idx < MAX_STABLE_PARAMS; param_idx++) {
			double param_avg = gsl_stats_mean(settings->param_values[comp_idx][param_idx] + settings->burnin_period, 1, num_valid_iter);
			double param_std = gsl_stats_sd(settings->param_values[comp_idx][param_idx] + settings->burnin_period, 1, num_valid_iter);

			if (settings->fix_components_during_last_n_iterations > 0 && (param_avg != 0 && param_std / fabs(param_avg) > 0.01)) {
				size_t check_only_last = settings->fix_components_during_last_n_iterations;
				size_t last_valid = check_only_last < num_valid_iter ? num_valid_iter - check_only_last : 0;
				param_avg = gsl_stats_mean(settings->param_values[comp_idx][param_idx] + last_valid + settings->burnin_period, 1, check_only_last);
				param_std = gsl_stats_sd(settings->param_values[comp_idx][param_idx] + last_valid + settings->burnin_period, 1, check_only_last);
			}

			settings->final_param_avg[comp_idx][param_idx] = param_avg;
			settings->final_param_std[comp_idx][param_idx] = param_std;
			new_params[param_idx] = param_avg;
		}

		double weight_avg = gsl_stats_mean(settings->weights[comp_idx] + settings->burnin_period, 1, num_valid_iter);
		double weight_sd = gsl_stats_sd(settings->weights[comp_idx] + settings->burnin_period, 1, num_valid_iter);

		dist->mixture_weights[comp_idx] = weight_avg;
		stable_setparams_array(dist->mixture_components[comp_idx], new_params);

		settings->final_weight_avg[comp_idx] = weight_avg;
		settings->final_weight_std[comp_idx] = weight_sd;
	}

	settings->ks_test = stable_kolmogorov_smirnov_gof(dist, data, length);

	for (comp_idx = 0; comp_idx < dist->num_mixture_components; comp_idx++) {
		for (param_idx = 0; param_idx < MAX_STABLE_PARAMS; param_idx++) {
			for (size_t param2_idx = 0; param2_idx < MAX_STABLE_PARAMS; param2_idx++) {
				settings->correlations[comp_idx][param_idx][param2_idx] = gsl_stats_correlation(
							settings->param_values[comp_idx][param_idx] + settings->burnin_period, 1,
							settings->param_values[comp_idx][param2_idx] + settings->burnin_period, 1, num_valid_iter);

			}
		}
	}
}

void stable_fit_mixture_print_results(struct stable_mcmc_settings* settings)
{
	size_t comp_idx, param_idx;

	printf("Mixture estimation results:\n");
	printf("Component |      α -  std  |      β -  std  |      μ -  std  |      σ -  std  | weight -  std  \n");

	for (comp_idx = 0; comp_idx < settings->num_final_components; comp_idx++) {
		printf("%9zu", comp_idx);

		for (param_idx = 0; param_idx < MAX_STABLE_PARAMS; param_idx++) {
			double param_avg = settings->final_param_avg[comp_idx][param_idx];
			double param_std = settings->final_param_std[comp_idx][param_idx];

			printf(" | %6.2lf - %5.2lf", param_avg, param_std);
		}

		double weight_avg = settings->final_weight_avg[comp_idx];
		double weight_std = settings->final_weight_std[comp_idx];

		printf(" | %6.2lf - %5.2lf\n", weight_avg, weight_std);
	}

	printf("\nKS test: %lf\n", settings->ks_test);
	printf("Acceptance ratio: %.3lf\n", settings->acceptance_ratio);
	printf("Correlations: \n");

	for (comp_idx = 0; comp_idx < settings->num_final_components; comp_idx++) {
		printf("Component %zu:\n", comp_idx);

		printf("   | %7s | %7s | %7s | %7s\n", "α", "β", "μ", "σ");

		for (param_idx = 0; param_idx < MAX_STABLE_PARAMS; param_idx++) {
			printf(" %s ", _param_names[param_idx]);

			for (size_t param2_idx = 0; param2_idx < MAX_STABLE_PARAMS; param2_idx++) {
				if (param2_idx == param_idx)
					printf("| %7s", " ");
				else {
					double corr = settings->correlations[comp_idx][param_idx][param2_idx];
					printf("| %6.3lf ", corr);
				}
			}

			printf("\n");
		}

	}

}
