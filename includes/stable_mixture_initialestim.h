/**
 * Header file for the interface of the mixture initial estimator.
 */

#ifndef STABLE_MIXTURE_INITIALESTIM
#define STABLE_MIXTURE_INITIALESTIM

#include "stable_api.h"

/**
 * Given a set of data, prepare an initial estimation of the components of the mixture.
 * @param dist   StableDist structure.
 * @param data   Data to estimate.
 * @param length Length of the data array.
 */
void stable_mixture_prepare_initial_estimation(
    StableDist* dist, const double* data, const unsigned int length, struct stable_mcmc_settings* settings);

#endif
