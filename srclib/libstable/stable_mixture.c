#include "stable_api.h"
#include "kde.h"
#include "stable_gridfit.h"
#include "methods.h"

#include <signal.h>
#include <unistd.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_gamma.h>

#define MAX_MIXTURE_ITERATIONS 10000
#define BURNIN_PERIOD 500
#define NUM_ALTERNATIVES_PARAMETER 1 // Number of alternative parameter values considered.

#define DO_VARIABLE_COMPONENTS
#define DO_WEIGHT_ESTIMATION
// #define DECREMENT_GENERATION_VARIANCE

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

static double _draw_rand(StableDist* dist, double mn, double mx, double mean, double std)
{
	double rnd = gsl_ran_gaussian(dist->gslrand, std) + mean;

	return min(mx, max(mn, rnd));
}

static double _draw_rand_alpha(StableDist *dist)
{
	return _draw_rand(dist, 0.0, 2.0, dist->alfa, 0.05);
}

static double _draw_rand_beta(StableDist *dist)
{
	return _draw_rand(dist, -1, 1, dist->beta, 0.1);
}

static double _draw_rand_mu(StableDist *dist)
{
	return _draw_rand(dist, -DBL_MAX, DBL_MAX, dist->mu_0, 0.3);
}

static double _draw_rand_sigma(StableDist *dist)
{
	return _draw_rand(dist, 0, DBL_MAX, dist->sigma, 0.02);
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

static short _is_local_min(double* data, size_t pos)
{
	return data[pos] <= data[pos + 1] && data[pos] <= data[pos - 1];
}

static short _is_local_max(double* data, size_t pos)
{
	return data[pos] >= data[pos + 1] && data[pos] >= data[pos - 1];
}

// Magic estimation based on EPDF data. See the paper for the reasoning.
static double _do_alpha_estim(double sep_logratio, double asym_log)
{
	double offset = 0.2021;
	double raw_alpha_estim = 1 / (7.367 * pow(sep_logratio - 0.8314, 0.7464));
	double asym_correction_factor = exp(- 1.081 * pow(asym_log * asym_log, 0.8103));

	return max(0.3, min(offset + asym_correction_factor * raw_alpha_estim, 2));
}

static double _do_beta_estim(double alpha, double asym_log)
{
	if (alpha > 1.6 || isnan(asym_log))
		return 0;

	double alpha_factor = exp(-1.224 * (alpha + 1.959)) - exp(-1.224 * 3.959);
	double estim = 0.4394 * log(1 / (0.5 - 0.01968 * asym_log / alpha_factor) - 1);

	if (isnan(estim))
		return 0;

	return max(-1, min(1, estim));
}

static double _do_sigma_estim(double alpha, double beta, double sep_95)
{
	double alpha2 = pow(alpha, 2);
	double beta2 = pow(beta, 2);
	double alpha3 = pow(alpha, 3);
	double beta3 = pow(beta, 3);
	double alpha4 = pow(alpha, 4);
	double alpha5 = pow(alpha, 5);

	// Magical fitting polynomial. As alpha, beta are bounded, we don't have any problems.
	double expected_sep_sigma1 =
		0.3564 - 2.669 * alpha + 0.0001647 * beta
		+ 6.435 * alpha2 - 0.004488 * alpha * beta - 0.371 * beta2
		- 5.495 * alpha3 + 0.007918 * alpha2 * beta + 1.62 * alpha * beta2 + 0.0002298 * beta3
		+ 2.166 * alpha4 - 0.005782 * alpha3 * beta - 1.432 * alpha2 * beta2 + 0.004358 * alpha * beta3
		- 0.3302 * alpha5 + 0.001526 * alpha4 * beta + 0.3499 * alpha3 * beta2 - 0.002562 * alpha2 * beta3
		;

	printf("Expected sep is %lf\n", expected_sep_sigma1);

	double sigma_estim = sep_95 / expected_sep_sigma1;

	if (sigma_estim < 0) {
		printf("Warning: Sigma estimation is negative! (%lf)\n", sigma_estim);
		sigma_estim = 0.1;
	}

	return sigma_estim;
}

void _prepare_initial_estimation(StableDist* dist, const double* data, const unsigned int length)
{
	size_t epdf_points = 5000;
	double samples[length];
	double epdf_x[epdf_points];
	double epdf[epdf_points];
	double epdf_start, epdf_end, epdf_step;
	size_t maxs[length], mins[length], valid_max[length], valid_min[length];
	size_t max_idx = 0, min_idx = 0, total_max;
	size_t i;
	double minmax_coef_threshold = 0.8;
	short searching_min = 0, searching_max = 1; // Assume we're starting at a minimum.
	double max_value = - DBL_MAX;
	size_t current_lowest_min_pos;
	size_t extra_components = 0;
	double mu_values[dist->max_mixture_components];
	double sigma_values[dist->max_mixture_components];
	StableDist* comp;

	memcpy(samples, data, sizeof(double) * length);

	dist->birth_probs = calloc(dist->max_mixture_components, sizeof(double));
	dist->death_probs = calloc(dist->max_mixture_components, sizeof(double));

	printf("Begin initial estimation\n");

	gsl_sort(samples, 1, length);

	epdf_start = gsl_stats_quantile_from_sorted_data(samples, 1, length, 0.02);
	epdf_end = gsl_stats_quantile_from_sorted_data(samples, 1, length, 0.98);

	epdf_step = (epdf_end - epdf_start) / epdf_points;

	printf("Study range is [%lf, %lf], %zu points with step %lf\n", epdf_start, epdf_end, epdf_points, epdf_step);

	for (i = 0; i < epdf_points; i++) {
		epdf_x[i] = epdf_start + i * epdf_step;
		epdf[i] = kerneldensity(samples, epdf_x[i], length, 0.4); // Silverman's bandwidth estimator is too high for skewed, multimodal distributions.

		// Assume we start at a minimum
		if (i == 0) {
			mins[0] = 0;
			min_idx++;

			searching_max = 1;
			searching_min = 0;
		} else if (i > 1 && i < length - 1) {
			if (searching_max && epdf[i - 1] > 0.01 && _is_local_max(epdf, i - 1)) {
				if (epdf[i - 1] * minmax_coef_threshold > epdf[mins[min_idx - 1]]) {
					// If this is a big enough maximum, mark it and start searching
					// for the next minimum
					searching_max = 0;
					searching_min = 1;

					printf("Found max %zu at %lf = %lf\n", max_idx, epdf_x[i - 1], epdf[i - 1]);
					maxs[max_idx] = i - 1;
					max_idx++;

				}
			} else if (searching_min && _is_local_min(epdf, i - 1)) {
				double minmax_ratio = epdf[maxs[max_idx - 1]] / epdf[i - 1];

				if (minmax_ratio < minmax_coef_threshold) {
					max_idx--;  // If the difference with the previous max is not big enough, cancel the previous maximum
					printf("Max discard\n");

					// Still, mark it as a possible component
					extra_components++;
				} else {
					// If the difference is good enough, mark it as a minimum.
					mins[min_idx] = i - 1;
					printf("Found min %zu at %lf = %lf\n", max_idx, epdf_x[i - 1], epdf[i - 1]);
					min_idx++;
				}

				// In any case, we need to search for the maximum: if the previous was suppressed we need
				// a new one; and if we found a minimum we also need another one.
				searching_max = 1;
				searching_min = 0;
			}
		} else if (i == length - 1 && searching_min) {
			// Mark a minimum at the end of the data if we are searching for one.
			mins[min_idx] = i;
			min_idx++;
		}

		if (epdf[i] > max_value)
			max_value = epdf[i];
	}

	total_max = max_idx;
	max_idx = 0;
	current_lowest_min_pos = mins[0];

	// The initial search of maximums is complete.
	// Now let's filter and get only those big enough
	for (i = 0; i < total_max; i++) {
		if (epdf[current_lowest_min_pos] > epdf[mins[i]])
			current_lowest_min_pos = mins[i];

		if (epdf[maxs[i]] > 0.25 * max_value) {
			valid_max[max_idx] = maxs[i];
			valid_min[max_idx] = current_lowest_min_pos;
			current_lowest_min_pos = maxs[i]; // Find minimum from here.
			max_idx++;
			printf("Found valid max %zu at %lf = %lf\n", max_idx, epdf_x[maxs[i]], epdf[maxs[i]]);
		} else
			extra_components++;
	}

	valid_min[max_idx] = mins[i]; // Add the last minimum (there must be n max, n + 1 mins).

	total_max = max_idx;

	stable_set_mixture_components(dist, total_max);

	for (i = 0; i < dist->num_mixture_components; i++) {
		comp = dist->mixture_components[i];

		size_t comp_begin = valid_min[i];
		size_t comp_end = valid_min[i + 1];
		size_t max_pos = valid_max[i];
		double max_value = epdf[valid_max[i]];
		size_t pos;

		printf("Initial C%zu: [%zu:%zu] (%lf:%lf)\n", i, comp_begin, comp_end, epdf_x[comp_begin], epdf_x[comp_end]);

		double left_deriv_95 = get_derivative_at_pctg_of_max(epdf + comp_begin, max_pos - comp_begin, max_value, 0.95, epdf_step, 1, &pos);
		double left_x_95 = epdf_x[pos + comp_begin];
		double right_deriv_95 = get_derivative_at_pctg_of_max(epdf + max_pos, comp_end - max_pos, max_value, 0.95, epdf_step, 0, &pos);
		double right_x_95 = epdf_x[pos + max_pos];

		printf("95 %% values: %lf at %lf, %lf at %lf\n", left_deriv_95, left_x_95, right_deriv_95, right_x_95);

		double left_deriv_75 = get_derivative_at_pctg_of_max(epdf + comp_begin, max_pos - comp_begin, max_value, 0.75, epdf_step, 1, &pos);
		double left_x_75 = epdf_x[pos + comp_begin];
		double right_deriv_75 = get_derivative_at_pctg_of_max(epdf + max_pos, comp_end - max_pos, max_value, 0.75, epdf_step, 0, &pos);
		double right_x_75 = epdf_x[pos + max_pos];

		printf("75 %% values: %lf at %lf, %lf at %lf\n", left_deriv_75, left_x_75, right_deriv_75, right_x_75);

		double sep_95 = right_x_95 - left_x_95;
		double sep_75 = right_x_75 - left_x_75;

		double sep_logratio = log(sep_75) - log(sep_95);
		double asym_log = log(left_deriv_95) - log(-right_deriv_95);

		printf("Estimation parameters: %lf %lf\n", sep_logratio, asym_log);

		comp->alfa = _do_alpha_estim(sep_logratio, asym_log);
		comp->beta = _do_beta_estim(comp->alfa, asym_log);
		comp->sigma = _do_sigma_estim(comp->alfa, comp->beta, sep_95);
		comp->mu_0 = epdf_x[max_pos];

		mu_values[i] = comp->mu_0;
		sigma_values[i] = comp->sigma;

		printf("C%zu initial %lf %lf %lf %lf\n", i, comp->alfa, comp->beta, comp->mu_0, comp->sigma);
	}

	printf("Found %zu possible extra components\n", extra_components);

	// Configure the death/birth probabilities
	if (extra_components > 0) {
		printf("Configuring extra component probabilities\n");
		double last_birth_prob = 0;

		for (i = dist->num_mixture_components; i < dist->max_mixture_components; i++) {
			if (i >= dist->num_mixture_components + extra_components) {
				dist->birth_probs[i] = last_birth_prob / 2; // Marginal probability of increasing components
			} else {
				size_t extra_comp_idx = i - dist->num_mixture_components;
				dist->birth_probs[i] = min((((double)extra_components) / dist->max_mixture_components) * (extra_components - extra_comp_idx) / extra_components, 0.05);
				last_birth_prob = dist->birth_probs[i];
			}

			if (i == dist->num_mixture_components)
				dist->death_probs[i] = 0;
			else
				dist->death_probs[i] = 1 - dist->birth_probs[i];

			printf("Probabilities %zu: %lf / %lf\n", i, dist->birth_probs[i], dist->death_probs[i]);
		}
	}

	// Prepare the priors
	dist->prior_mu_avg = gsl_stats_mean(mu_values, 1, dist->num_mixture_components);
	dist->prior_mu_variance = gsl_stats_variance(mu_values, 1, dist->num_mixture_components);
	dist->prior_weights = 10; // TODO: ¿?
	double sigma_mean = gsl_stats_mean(sigma_values, 1, dist->num_mixture_components);
	double sigma_variance = gsl_stats_variance(sigma_values, 1, dist->num_mixture_components);
	dist->prior_sigma_alpha0 = pow(sigma_mean, 2) / sigma_variance + 2;
	dist->prior_sigma_beta0 = sigma_mean * (dist->prior_sigma_alpha0 - 1);
}

static void _do_component_split(
	StableDist * dist, size_t comp_to_split, double w1, double w2,
	double params_1[4], double params_2[4])
{
	size_t split_1 = comp_to_split;
	size_t split_2 = dist->num_mixture_components; // Component 2 is created at the last position

	stable_set_mixture_components(dist, dist->num_mixture_components + 1);

	dist->mixture_weights[split_1] = w1;
	dist->mixture_weights[split_2] = w2;

	stable_setparams_array(dist->mixture_components[split_1], params_1);
	stable_setparams_array(dist->mixture_components[split_2], params_2);
}

static void _do_component_combine(
	StableDist * dist, size_t comp_1, size_t comp_2, double w_comb,
	double params[4])
{
	size_t combined_comp = min(comp_1, comp_2);
	size_t removed_comp = max(comp_1, comp_2);

	if (removed_comp < dist->num_mixture_components - 1) {
		// Not the last component, replace this one with the last
		StableDist* swap = dist->mixture_components[dist->num_mixture_components - 1];
		dist->mixture_components[dist->num_mixture_components - 1] = dist->mixture_components[removed_comp];
		dist->mixture_components[removed_comp] = swap;
	}

	stable_set_mixture_components(dist, dist->num_mixture_components - 1);
	dist->mixture_weights[combined_comp] = w_comb;
	stable_setparams_array(dist->mixture_components[combined_comp], params);
}

static short _calc_splitcombine_acceptance_ratio(
	StableDist * dist, const double * data, const unsigned int length, short is_split, double * current_pdf,
	size_t comp_1, size_t comp_2,
	double w1, double w2, double w_comb, double u1, double u2, double u3,
	double params_1[4], double params_2[4], double params_comb[4])
{
	double new_pdf[length];

	if (is_split) {
		_do_component_split(dist, comp_1, w1, w2, params_1, params_2);
		comp_2 = dist->num_mixture_components - 1; // New component is the last one
	} else
		_do_component_combine(dist, comp_1, comp_2, w_comb, params_comb);

	stable_pdf_gpu(dist, data, length, new_pdf, NULL);

	double n1 = length * w1;
	double n2 = length * w2;

	double mu1 = params_1[STABLE_PARAM_MU], mu2 = params_2[STABLE_PARAM_MU], mu_comb = params_comb[STABLE_PARAM_MU];
	double sigma1 = params_1[STABLE_PARAM_SIGMA], sigma2 = params_2[STABLE_PARAM_SIGMA], sigma_comb = params_comb[STABLE_PARAM_SIGMA];

	double likelihood_ratio = 1;

	for (size_t k = 0; k < length; k++)
		likelihood_ratio *= new_pdf[k] / current_pdf[k];

	double alpha_beta_ratio = 0.5;
	double weight_ratio =
		(
			pow(w1, dist->prior_weights - 1 + n1)
			* pow(w2, dist->prior_weights - 1 + n2)
		) / (
			pow(w_comb, dist->prior_weights - 1 + n1 + n2)
			* gsl_sf_beta(dist->prior_weights, dist->num_mixture_components * dist->prior_weights)
		);
	double mu_ratio =
		sqrt(1 / (M_2_PI * dist->prior_mu_variance))
		* exp(-0.5 * (1 / dist->prior_mu_variance) * (
				  pow(mu1 - dist->prior_mu_avg, 2) + pow(mu2 - dist->prior_mu_avg, 2) - pow(mu_comb - dist->prior_mu_avg, 2)
			  ));

	double sigma_ratio =
		(pow(dist->prior_sigma_beta0, dist->prior_sigma_alpha0) / gsl_sf_gamma(dist->prior_sigma_alpha0))
		* pow(sigma1 * sigma2 / sigma_comb, 2 * (-dist->prior_sigma_alpha0 - 1))
		* exp(- dist->prior_sigma_beta0 * (pow(sigma1, -2) + pow(sigma2, -2) - pow(sigma_comb, -2)));

	double move_probability = 1 / (gsl_ran_beta_pdf(u1, 2, 2) * gsl_ran_beta_pdf(u2, 2, 2) * gsl_ran_beta_pdf(u3, 1, 1));

	double jacobian =
		(w_comb * fabs(mu1 - mu2) * pow(sigma1, 2) * pow(sigma2, 2))
		/ (u2 * (1 - pow(u2, 2)) * (1 - u3) * pow(sigma_comb, 2));

	double acceptance_ratio =
		likelihood_ratio * alpha_beta_ratio * weight_ratio *
		mu_ratio * sigma_ratio * move_probability * jacobian;

	if (!is_split)
		acceptance_ratio = 1 / acceptance_ratio;

	printf("Ratios: αβ = %lf, w = %lf, μ = %lf, σ = %lf, m = %lf, j = %lf\n", alpha_beta_ratio, weight_ratio, mu_ratio, sigma_ratio, move_probability, jacobian);
	printf("Acceptance ratio: %lf\n", acceptance_ratio);

	if (rand_event(dist->gslrand, acceptance_ratio))
		return 1; // Move accepted

	// If move is not accepted, revert the previous operation

	if (is_split)
		_do_component_combine(dist, comp_1, comp_2, w_comb, params_comb);
	else
		_do_component_split(dist, comp_1, w1, w2, params_1, params_2);

	return 0;
}

static int _check_split_move(StableDist * dist, const double * data, const unsigned int length, double * current_pdf)
{
	size_t comp_idx;
	short accepted;

	if (dist->num_mixture_components == dist->max_mixture_components)
		return 0;

	for (comp_idx = 0; comp_idx < dist->num_mixture_components; comp_idx++) {
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
			continue;

		double new_sigma_1 = sqrt(u3 * (1 - pow(u2, 2)) * comp->sigma * curr_weight / new_weight_1);
		double new_sigma_2 = sqrt((1 - u3) * (1 - pow(u2, 2)) * comp->sigma * curr_weight / new_weight_2);

		double params_1[4] = { curr_params[STABLE_PARAM_ALPHA], curr_params[STABLE_PARAM_BETA], new_mu_1, new_sigma_1 };
		double params_2[4] = { curr_params[STABLE_PARAM_ALPHA], curr_params[STABLE_PARAM_BETA], new_mu_2, new_sigma_2 };

		accepted = _calc_splitcombine_acceptance_ratio(
					   dist, data, length, 1, current_pdf,
					   comp_idx, -1, new_weight_1, new_weight_2, curr_weight, u1, u2, u3,
					   params_1, params_2, curr_params);

		return accepted; // Only 1 proposal
	}

	return 0;
}

static size_t _mixtures_with_mu_in_range(StableDist * dist, double mu_min, double mu_max)
{
	size_t num_comps = 0;

	for (size_t i = 0; i < dist->num_mixture_components; i++) {
		if (dist->mixture_components[i]->mu_0 > mu_min && dist->mixture_components[i]->mu_0 < mu_max)
			num_comps++;
	}

	return num_comps;
}

static int _check_combine_move(StableDist * dist, const double * data, const unsigned int length, double * current_pdf)
{
	size_t comp_1_idx, comp_2_idx;
	StableDist* comp_1, *comp_2;
	short accepted;

	for (comp_1_idx = 0; comp_1_idx < dist->num_mixture_components - 1; comp_1_idx++) {
		for (comp_2_idx = comp_1_idx + 1; comp_2_idx < dist->num_mixture_components; comp_2_idx++) {
			comp_1 = dist->mixture_components[comp_1_idx];
			comp_2 = dist->mixture_components[comp_2_idx];

			double params_1[4], params_2[4], params_comb[4];
			stable_getparams_array(comp_1, params_1);
			stable_getparams_array(comp_2, params_2);

			double mu_min = min(comp_1->mu_0, comp_2->mu_0);
			double mu_max = max(comp_1->mu_0, comp_2->mu_0);

			if (_mixtures_with_mu_in_range(dist, mu_min, mu_max) > 0)
				continue;

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

			accepted = _calc_splitcombine_acceptance_ratio(
						   dist, data, length, 0, current_pdf,
						   comp_1_idx, comp_2_idx, w1, w2, w_comb, u1, u2, u3,
						   params_1, params_2, params_comb);

			return accepted; // Only 1 proposal
		}
	}

	return 0;
}

int stable_fit_mixture(StableDist * dist, const double * data, const unsigned int length)
{
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
	double dirichlet_params[dist->max_mixture_components];
	double param_probability;
	double previous_weights[dist->max_mixture_components];
	double param_values[dist->max_mixture_components][MAX_STABLE_PARAMS][MAX_MIXTURE_ITERATIONS];
	double weights[dist->max_mixture_components][MAX_MIXTURE_ITERATIONS];

	FILE* debug_data = fopen("mixture_debug.dat", "w");

	_prepare_initial_estimation(dist, data, length);

	for (i = 0; i < dist->num_mixture_components; i++) {
		// TODO: More magic.
#ifdef DO_WEIGHT_ESTIMATION
		dist->mixture_weights[i] = ((double) 1) / dist->num_mixture_components;
#endif
		dist->mixture_components[i]->mixture_montecarlo_variance = 0.05;

		previous_param_probs[i] = 1;
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

	for (i = 0; i < BURNIN_PERIOD + MAX_MIXTURE_ITERATIONS && !stop; i++) {
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

					if (i >= BURNIN_PERIOD)
						param_values[comp_idx][param_idx][i - BURNIN_PERIOD] = new_params[param_idx];
				}
			}
		}

		// Estimate the weights. TODO: Not completely sure of this.
#ifdef DO_WEIGHT_ESTIMATION
		memcpy(previous_weights, dist->mixture_weights, dist->max_mixture_components * sizeof(double));

		for (comp_idx = 0; comp_idx < dist->num_mixture_components; comp_idx++)
			dirichlet_params[comp_idx] = dist->prior_weights + length * dist->mixture_weights[comp_idx];

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


		if (i >= BURNIN_PERIOD) {
			for (comp_idx = 0; comp_idx < dist->num_mixture_components; comp_idx++)
				weights[comp_idx][i - BURNIN_PERIOD] = dist->mixture_weights[comp_idx];
		}

#endif

#ifdef DO_VARIABLE_COMPONENTS

		if (rand_event(dist->gslrand, dist->birth_probs[dist->num_mixture_components])) {
			printf("\nTry split\n");

			if (_check_split_move(dist, data, length, previous_pdf)) {
				num_changes++;
				printf("Accept split\n");
			}
		} else if (rand_event(dist->gslrand, dist->death_probs[dist->num_mixture_components])) {
			printf("\nTry combine\n");

			if (_check_combine_move(dist, data, length, previous_pdf)) {
				num_changes++;
				printf("Accept combine\n");
			}
		}

#endif

		if (num_changes == 0)
			streak_without_change++;
		else
			streak_without_change = 0;

		printf("\rIter %zu - %zu changes", i, num_changes);
		fflush(stdout);
		fprintf(debug_data, "%zu %zu", num_changes, dist->num_mixture_components);

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
#ifdef DECREMENT_GENERATION_VARIANCE
			component->mixture_montecarlo_variance *= 0.99999;
#endif
		}
	}


	printf("\n");

	if (i > BURNIN_PERIOD) {
		printf("Mixture estimation results:\n");
		printf("Component |      α -  std  |      β -  std  |      μ -  std  |      σ -  std  | weight -  std  \n");

		for (comp_idx = 0; comp_idx < dist->num_mixture_components; comp_idx++) {
			printf("%9zu", comp_idx);

			for (param_idx = 0; param_idx < MAX_STABLE_PARAMS; param_idx++) {
				double param_avg = gsl_stats_mean(param_values[comp_idx][param_idx], 1, i - BURNIN_PERIOD);
				double param_sd = gsl_stats_sd_m(param_values[comp_idx][param_idx], 1, i - BURNIN_PERIOD, param_avg);

				printf(" | %6.2lf - %5.2lf", param_avg, param_sd);
				new_params[param_idx] = param_avg;
			}

			double weight_avg = gsl_stats_mean(weights[comp_idx], 1, i - BURNIN_PERIOD);
			double weight_sd = gsl_stats_mean(weights[comp_idx], 1, i - BURNIN_PERIOD);

			dist->mixture_weights[comp_idx] = weight_avg;

			printf(" | %6.2lf - %5.2lf", weight_avg, weight_sd);

			stable_setparams_array(dist->mixture_components[comp_idx], new_params);
			printf("\n");
		}
	} else
		printf("WARNING: Burn-in period not passed.\n");

#ifdef DEBUG
	fclose(debug_data);
#endif

	return 0;
}
