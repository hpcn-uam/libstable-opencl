#include "stable_api.h"
#include "methods.h"

#include <unistd.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_gamma.h>

double MIXTURE_KERNEL_ADJUST = 0.9;
double MIXTURE_KERNEL_ADJUST_FINER = 0.12 * 0.9;


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

	if (sigma_estim <= 0) {
		printf("Warning: Sigma estimation is non-positive! (%lf)\n", sigma_estim);
		sigma_estim = 0.1;
	}

	return sigma_estim;
}

static void _component_initial_estimation(StableDist* comp, double start_x, double end_x, double* epdf, double* epdf_x, size_t epdf_points)
{
	double epdf_step = (end_x - start_x) / epdf_points;
	size_t max_pos;
	double max_value;
	size_t pos;

	// Recalculate maximum point with the EPDF
	max_pos = gsl_stats_max_index(epdf, 1, epdf_points);
	max_value = epdf[max_pos];

	double left_deriv_95 = get_derivative_at_pctg_of_max(epdf, max_pos, max_value, 0.95, epdf_step, 1, &pos);

	double left_x_95 = epdf_x[pos];
	double right_deriv_95 = get_derivative_at_pctg_of_max(epdf + max_pos, epdf_points - max_pos, max_value, 0.95, epdf_step, 0, &pos);
	double right_x_95 = epdf_x[pos + max_pos];

	printf("95 %% values: %lf at %lf, %lf at %lf\n", left_deriv_95, left_x_95, right_deriv_95, right_x_95);

	double left_deriv_75 = get_derivative_at_pctg_of_max(epdf, max_pos, max_value, 0.75, epdf_step, 1, &pos);
	double left_x_75 = epdf_x[pos];
	double right_deriv_75 = get_derivative_at_pctg_of_max(epdf + max_pos, epdf_points - max_pos, max_value, 0.75, epdf_step, 0, &pos);
	double right_x_75 = epdf_x[pos + max_pos];

	printf("75 %% values: %lf at %lf, %lf at %lf\n", left_deriv_75, left_x_75, right_deriv_75, right_x_75);

	double sep_95 = right_x_95 - left_x_95;
	double sep_75 = right_x_75 - left_x_75;

	if (-left_deriv_95 / right_deriv_95 <= 0)
		printf("WARNING: weird (negative) value for assymetry.\n");

	double sep_logratio = log(sep_75) - log(sep_95);
	double asym_log = log(left_deriv_95) - log(-right_deriv_95);

	printf("Estimation parameters: %lf %lf\n", sep_logratio, asym_log);

	comp->alfa = _do_alpha_estim(sep_logratio, asym_log);
	comp->beta = _do_beta_estim(comp->alfa, asym_log);
	comp->sigma = _do_sigma_estim(comp->alfa, comp->beta, sep_95);
	comp->mu_0 = epdf_x[max_pos];
}

size_t _find_local_minmax(double* epdf, size_t* maxs, size_t* mins, size_t epdf_points, double* max_value, double* epdf_x)
{
	size_t i;
	size_t min_idx = 0, max_idx = 0;
	short searching_min = 0, searching_max = 1; // Assume we're starting at a minimum.
	double minmax_coef_threshold = 0.9;

	if (max_value != NULL)
		*max_value = -DBL_MAX;

	for (i = 0; i < epdf_points; i++) {
		// Assume we start at a minimum
		if (i == 0) {
			mins[0] = 0;
			min_idx++;

			searching_max = 1;
			searching_min = 0;
		} else if (i > 1 && i < epdf_points - 1) {
			if (searching_max && epdf[i - 1] > 0.01 && _is_local_max(epdf, i - 1)) {
				double minmax_ratio = epdf[mins[min_idx - 1]] / epdf[i - 1];

				if (minmax_ratio < minmax_coef_threshold) {
					// If this is a big enough maximum, mark it and start searching
					// for the next minimum
					searching_max = 0;
					searching_min = 1;

					printf("Found max %zu at %lf = %lf (ratio %lf)\n", max_idx, epdf_x[i - 1], epdf[i - 1], minmax_ratio);
					maxs[max_idx] = i - 1;
					max_idx++;

				} else
					printf("Max discard at %lf (ratio %lf)\n", epdf_x[i - 1], minmax_ratio);
			} else if (searching_min && _is_local_min(epdf, i - 1)) {
				double minmax_ratio = epdf[i - 1] / epdf[maxs[max_idx - 1]];

				if (minmax_ratio > minmax_coef_threshold) {
					max_idx--;  // If the difference with the previous max is not big enough, cancel the previous maximum
					printf("Low min at %lf, discard previous max (ratio %lf)\n", epdf_x[i - 1], minmax_ratio);
				} else {
					// If the difference is good enough, mark it as a minimum.
					mins[min_idx] = i - 1;
					printf("Found min %zu at %lf = %lf (ratio %lf)\n", max_idx, epdf_x[i - 1], epdf[i - 1], minmax_ratio);
					min_idx++;
				}

				// In any case, we need to search for the maximum: if the previous was suppressed we need
				// a new one; and if we found a minimum we also need another one.
				searching_max = 1;
				searching_min = 0;
			}
		} else if (i == epdf_points - 1 && searching_min) {
			// Mark a minimum at the end of the data if we are searching for one.
			mins[min_idx] = i;
			min_idx++;
		}

		if (max_value != NULL && epdf[i] > *max_value)
			*max_value = epdf[i];
	}

	return max_idx;
}

struct component_partition {
	size_t begin_idx;
	size_t end_idx;
	size_t max_idx;
	short found_in_finer_epdf;
};

static int _compar_partition(const void* a, const void* b)
{
	const struct component_partition* pa = (const struct component_partition*) a;
	const struct component_partition* pb = (const struct component_partition*) b;

	// Cast to int to allow signed results
	return ((int) pa->begin_idx) - ((int) pb->begin_idx);
}

void stable_mixture_prepare_initial_estimation(StableDist* dist, const double* data, const unsigned int length)
{
	size_t epdf_points = 5000;
	double samples[length];
	double epdf_x[epdf_points];
	double epdf[epdf_points], epdf_finer[epdf_points];
	double epdf_start, epdf_end;
	double min_peak_dst;
	size_t maxs[epdf_points], mins[epdf_points];
	size_t maxs_finer[epdf_points], mins_finer[epdf_points];
	size_t total_max, total_max_finer, max_idx, max_finer_idx;
	struct component_partition mixture_partition[dist->max_mixture_components];
	size_t current_partition, total_partitions;
	size_t i;
	double max_value;
	size_t current_lowest_min_pos;
	size_t extra_components = 0;
	size_t extra_partitions = 0;
	double mu_values[dist->max_mixture_components];
	double sigma_values[dist->max_mixture_components];
	StableDist* comp;
	size_t data_offset = 0;

	memcpy(samples, data, sizeof(double) * length);

	dist->birth_probs = calloc(dist->max_mixture_components, sizeof(double));
	dist->death_probs = calloc(dist->max_mixture_components, sizeof(double));

	printf("Begin initial estimation\n");

	gsl_sort(samples, 1, length);

	epdf_start = gsl_stats_quantile_from_sorted_data(samples, 1, length, 0.02);
	epdf_end = gsl_stats_quantile_from_sorted_data(samples, 1, length, 0.98);

	min_peak_dst = 0.05 * (epdf_end - epdf_start);

	printf("Study range is [%lf, %lf], %zu points with step %lf\n", epdf_start, epdf_end, epdf_points, (epdf_end - epdf_end) / epdf_points);

	calculate_epdf(samples, length, epdf_start, epdf_end, epdf_points, MIXTURE_KERNEL_ADJUST, epdf_x, epdf);
	calculate_epdf(samples, length, epdf_start, epdf_end, epdf_points, MIXTURE_KERNEL_ADJUST_FINER, epdf_x, epdf_finer);

	total_max = _find_local_minmax(epdf, maxs, mins, epdf_points, &max_value, epdf_x);
	total_max_finer = _find_local_minmax(epdf_finer, maxs_finer, mins_finer, epdf_points, NULL, epdf_x);

	current_lowest_min_pos = mins[0];

	// The initial search of maximums is complete.
	// Now let's filter and get only those big enough in the smoother EPDF
	// This allows us to get a good picture of where are the maximums of the major
	// mixture components without having to worry much about noise
	for (current_partition = 0, max_idx = 0; max_idx < total_max; max_idx++) {
		if (epdf[current_lowest_min_pos] > epdf[mins[max_idx]])
			current_lowest_min_pos = mins[max_idx];

		if (epdf[maxs[max_idx]] > 0.05 * max_value) {
			mixture_partition[current_partition].begin_idx = current_lowest_min_pos;
			mixture_partition[current_partition].max_idx = maxs[max_idx];
			mixture_partition[current_partition].found_in_finer_epdf = 0;

			if (current_partition > 0)
				mixture_partition[current_partition - 1].end_idx = current_lowest_min_pos;

			current_lowest_min_pos = maxs[max_idx]; // Find minimum from here.
			printf("Max %zu at %lf = %lf is valid (assigned to component %zu)\n",
				   max_idx, epdf_x[maxs[max_idx]], epdf[maxs[max_idx]], current_partition);

			current_partition++;
		} else {
			extra_components++;
			printf("Max %zu discarded (%.3lf %% of max)\n", max_idx, 100 * epdf[maxs[max_idx]] / max_value);
		}
	}

	mixture_partition[current_partition - 1].end_idx = mins[max_idx]; // Add the last minimum (there must be n max, n + 1 mins).
	total_partitions = current_partition;

	// Now a second search on the set of local min/max obtained from the finer
	// EPDF estimation. This one has much more noise, but it does not smooth out
	// peaks that could signal the existence of another mixture. In this case,
	// we will only select peaks differing significantly from their value in the
	// smoother EPDF.
	for (max_idx = 0; max_idx < total_max_finer; max_idx++) {
		size_t max_pos = maxs_finer[max_idx];
		double difference_ratio = epdf_finer[max_pos] / epdf[max_pos];

		if (difference_ratio > 1.7 && epdf[max_pos] > 0.05 * max_value) {
			// Found a peak that grows considerably when compared to the smoother EPDF.
			// Most probably, this is a peak of a mixture, so we have to insert a new partition.

			// Search for the first partition which has a maximum with X value higher than the one
			// we're treating now
			size_t part_idx = 0;
			size_t minval_idx;
			size_t part_max;
			size_t part_end = epdf_points - 1;

			while (part_idx < total_partitions && epdf_x[mixture_partition[part_idx].max_idx] < epdf_x[max_pos])
				part_idx++;

			printf("Found extra peak ratio %lf at x[%zu] = %lf, comp after is %zu\n", difference_ratio, max_pos, epdf_x[max_pos], part_idx);

			if (part_idx > 0) {
				// Modify the partition before this one
				part_max = mixture_partition[part_idx - 1].max_idx;

				minval_idx = gsl_stats_min_index(epdf_finer + part_max, 1, max_pos - part_max) + part_max;
				mixture_partition[total_partitions].begin_idx = minval_idx;
				part_end = mixture_partition[part_idx - 1].end_idx;
				mixture_partition[part_idx - 1].end_idx = minval_idx;
			} else {
				// No partition before this one, the begin index of the new partition is the begin index
				// of the one after.
				mixture_partition[total_partitions].begin_idx = mixture_partition[part_idx].begin_idx;
			}

			if (part_idx < total_partitions) {
				// Modify the partition after this one
				part_max = mixture_partition[part_idx].max_idx;

				minval_idx = gsl_stats_min_index(epdf_finer + max_pos, 1, part_max - max_pos) + max_pos;
				mixture_partition[total_partitions].end_idx = minval_idx;
				mixture_partition[part_idx].begin_idx = minval_idx;
			} else {
				// No partition after this one, the end index of the new partition is the end
				// of the one before
				mixture_partition[total_partitions].end_idx = part_end;
			}

			mixture_partition[total_partitions].max_idx = max_pos;
			mixture_partition[total_partitions].found_in_finer_epdf = 1;

			total_partitions++;
			extra_partitions++;

			// Re-sort the partitions to allow search of the next maximum.
			// Use mergesort instead of qsort as it behaves better on almost-sorted data.
			mergesort(mixture_partition, total_partitions, sizeof(struct component_partition), _compar_partition);
		}
	}

	printf("Found %zu possible extra components, for a total of %zu\n", extra_partitions, total_partitions);
	stable_set_mixture_components(dist, total_partitions);

	for (i = 0; i < dist->num_mixture_components; i++) {
		comp = dist->mixture_components[i];

		size_t comp_begin = mixture_partition[i].begin_idx;
		size_t comp_end = mixture_partition[i].end_idx;
		size_t sample_len = 0;
		double* epdf_for_estimation = mixture_partition[i].found_in_finer_epdf ? epdf_finer : epdf;

		epdf_for_estimation += comp_begin;

		printf("Initial C%zu: [%zu:%zu] (%lf:%lf)\n", i, comp_begin, comp_end, epdf_x[comp_begin], epdf_x[comp_end]);
		_component_initial_estimation(comp, epdf_x[comp_begin], epdf_x[comp_end], epdf_for_estimation, epdf_x + comp_begin, comp_end - comp_begin);

		mu_values[i] = comp->mu_0;
		sigma_values[i] = comp->sigma;

		stable_print_params(comp, "C%zu parameters", i);

		data_offset = sample_len;
	}


	// Configure the death/birth probabilities
	printf("Configuring extra component probabilities\n");
	double last_birth_prob = 0.01;

	for (i = dist->num_mixture_components; i < dist->max_mixture_components; i++) {
		if (i >= dist->num_mixture_components + extra_components) {
			dist->birth_probs[i] = last_birth_prob / 2; // Marginal probability of increasing components
		} else {
			size_t extra_comp_idx = i - dist->num_mixture_components;
			dist->birth_probs[i] = 0.1 * max((((double)extra_components) / dist->max_mixture_components) * (extra_components - extra_comp_idx) / extra_components, 0.025);
		}

		last_birth_prob = dist->birth_probs[i];

		if (i == dist->num_mixture_components)
			dist->death_probs[i] = 0;
		else
			dist->death_probs[i] = 1 - dist->birth_probs[i];

		printf("Probabilities %zu: %lf / %lf\n", i, dist->birth_probs[i], dist->death_probs[i]);
	}

	// Prepare the priors
	dist->prior_mu_avg = gsl_stats_mean(mu_values, 1, dist->num_mixture_components);
	dist->prior_mu_variance = gsl_stats_variance(mu_values, 1, dist->num_mixture_components);
	dist->prior_weights = 10; // TODO: This does not look like it has any science on it.
	double sigma_mean = gsl_stats_mean(sigma_values, 1, dist->num_mixture_components);
	double sigma_variance = gsl_stats_variance(sigma_values, 1, dist->num_mixture_components);
	dist->prior_sigma_alpha0 = pow(sigma_mean, 2) / sigma_variance + 2;
	dist->prior_sigma_beta0 = sigma_mean * (dist->prior_sigma_alpha0 - 1);
}