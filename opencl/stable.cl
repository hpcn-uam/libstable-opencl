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

#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#warning "Double precision floating point not supported by OpenCL implementation."
#endif

#if defined(cl_nv_pragma_unroll)
#pragma OPENCL EXTENSION cl_nv_pragma_unroll : enable
#else
#warning "Loop unrolling is disabled"
#endif

#ifndef M_PI_2
#define M_PI_2     1.57079632679489661923132169163975144      // Pi/2
#endif

#ifndef M_PI
#define M_PI       3.14159265358979323846264338327950288       // Pi
#endif

#ifndef M_1_PI
#define M_1_PI 	 	0.318309886183790671537767526745028724 	   // 1 / Pi
#endif

#include "includes/opencl_common.h"
#include "includes/gk_points.h"
#include "includes/stable_inv_precalcs.h"

#define anyf(a) any((int2) a)
#define vec2(b) (cl_precision2)((b), (b))
#define vec4(b) (cl_precision4)((b), (b), (b), (b))
#define vec8(b) (cl_precision8)((b), (b), (b), (b), (b), (b), (b), (b))

#if POINTS_EVAL == 4
#define cl_vec cl_precision8
#define cl_halfvec cl_precision4
#define vec vec8
#define vech vec4
#elif POINTS_EVAL == 2
#define cl_vec cl_precision4
#define cl_halfvec cl_precision2
#define vec vec4
#define vech vec2
#elif POINTS_EVAL == 1
#define cl_vec cl_precision2
#define cl_halfvec cl_precision
#define vec vec2
#define vech
#endif

#define SUBINT_CONTRIB_TH 0.00001
#define MIN_CONTRIBUTING_SUBINTS GK_SUBDIVISIONS / 4

#define SET_TO_RESULT_AND_RETURN 1
#define CONTINUE_CALC 0


struct stable_precalc {
    cl_precision theta0_;
    cl_precision beta_;
    cl_precision xxipow;
    cl_precision ibegin;
    cl_precision iend;
    cl_precision subint_length;
    cl_precision xxi;
    cl_precision pdf_precalc;
    cl_precision cdf_precalc;
    cl_precision2 final_factor;
    cl_precision2 final_addition;
    size_t max_reevaluations;
};

// Just a debug macro to report to values out
// in gauss[point_index] and kronrod[point_index]
// and exit the kernel.
#define report_and_exit(g, k) do { \
	gauss[point_index] = g; \
	kronrod[point_index] = k; \
	return; \
} while (0)

// Evaluate the function in the corresponding Gauss-Kronrod POINT_EVAL points (i.e., in
// one, two or four points) and return a vector of 2 * POINT_EVAL values, where each pair
// is the Gauss and Kronrod results of the evaluation at the corresponding point.
cl_vec eval_gk_pair(constant struct stable_info* stable, struct stable_precalc* precalc)
{
	size_t gk_point = get_local_id(0);
	size_t subinterval_index = get_local_id(1);

	cl_precision2 w = gk_weights[gk_point];

	cl_halfvec centers = vech(precalc->ibegin + precalc->subint_length / 2); // Prepare the vector that defines the center of each interval.
	cl_precision abscissa = precalc->subint_length * gk_absc[gk_point] / 2; // Get the abscissa for this point, scale to our subinterval (half) length.
	cl_precision2 abscissa_vec = (cl_precision2)(- abscissa, abscissa); // GK quadrature is symmetric, we evaluate in +- abscissa.
	cl_halfvec subd_offsets = vech(subinterval_index); // Prepare the calculation of the offsets

	// Set the subinterval offsets. If we are evaluating, say, 2 points per interval, we have to
	// evaluate in the intervals indexed by subinterval_index and subinterval_index + MAX_WORKGROUPS.
	// We use vectors to reduce the number of operations.
#if POINTS_EVAL == 2
	subd_offsets += ((cl_halfvec)(0, 1)) * MAX_WORKGROUPS;
#elif POINTS_EVAL == 4
	subd_offsets += ((cl_halfvec)(0, 1, 2, 3)) * MAX_WORKGROUPS;
#endif

	// Scale the lengths and set the final centers of the intervals.
	subd_offsets *= precalc->subint_length;
	centers += subd_offsets;

	cl_vec cdf_val, pdf_val, final_val;

	// Calculate the points of evaluation: add each center to our two abscissae.
#if POINTS_EVAL == 1
	cdf_val.s01 = vec2(centers) + abscissa_vec;
#else
	cdf_val.s01 = centers.s00 + abscissa_vec;
#if POINTS_EVAL >= 2
	cdf_val.s23 = centers.s11 + abscissa_vec;
#if POINTS_EVAL >= 4
	cdf_val.s45 = centers.s22 + abscissa_vec;
	cdf_val.s67 = centers.s33 + abscissa_vec;
#endif
#endif
#endif

	// Now evaluate the function.
	cl_vec aux;
	cl_vec cosval = cos(cdf_val);

	// The difference between the calculation of the PDF and the CDF is just one
	// additional operation in the case of the PDF. We share the code and allow the
	// return of both values if required.
	if(is_integrand_eq1(stable->integrand))
	{
		aux = (precalc->beta_ * cdf_val + vec(M_PI_2)) / cosval;
		cdf_val = sin(cdf_val) * aux / precalc->beta_ + log(aux) + stable->k1;

		pdf_val = exp(cdf_val + precalc->xxipow);
		cdf_val = exp(-pdf_val);

		if(is_integrand_pdf(stable->integrand))
			pdf_val = cdf_val * pdf_val;
	}
	else if(is_integrand_neq1(stable->integrand))
	{
		aux = (precalc->theta0_ + cdf_val) * stable->alfa;
		cdf_val = log(cosval / sin(aux)) * stable->alfainvalfa1 +
			+ log(cos(aux - cdf_val) / cosval) + stable->k1;

		pdf_val = exp(cdf_val + precalc->xxipow);
		cdf_val = exp(-pdf_val);

		if(is_integrand_pdf(stable->integrand))
			pdf_val = cdf_val * pdf_val;
	}

	// GK quadrature is symmetric, so just add them in one quantity.
	// Just avoid the 0 (last evaluation point) because it's the only
	// point being evaluated once.
	if(gk_point < KRONROD_EVAL_POINTS - 1)
	{
		if(is_integrand_pdf(stable->integrand))
		{
			pdf_val.s0 += pdf_val.s1;
#if POINTS_EVAL >= 2
			pdf_val.s2 += pdf_val.s3;
#if POINTS_EVAL >= 4
			pdf_val.s4 += pdf_val.s5;
			pdf_val.s6 += pdf_val.s7;
#endif
#endif
		}
		if(is_integrand_cdf(stable->integrand))
		{
			cdf_val.s0 += cdf_val.s1;
#if POINTS_EVAL >= 2
			cdf_val.s2 += cdf_val.s3;
#if POINTS_EVAL >= 4
			cdf_val.s4 += cdf_val.s5;
			cdf_val.s6 += cdf_val.s7;
#endif
#endif
		}
	}

	// Now multiply the result by the corresponding weight and return.
	if(is_integrand_pcdf(stable->integrand))
	{
		final_val.s0 = w.s0 * pdf_val.s0;
		final_val.s1 = w.s0 * cdf_val.s0;
#if POINTS_EVAL >= 2
		final_val.s2 = w.s0 * pdf_val.s2;
		final_val.s3 = w.s0 * cdf_val.s2;
#if POINTS_EVAL >= 4
		final_val.s4 = w.s0 * pdf_val.s4;
		final_val.s5 = w.s0 * cdf_val.s4;
		final_val.s6 = w.s0 * pdf_val.s6;
		final_val.s7 = w.s0 * cdf_val.s6;
#endif
#endif
	}
	else if(is_integrand_cdf(stable->integrand))
	{
		final_val.s01 = w * cdf_val.s0;
#if POINTS_EVAL >= 2
		final_val.s23 = w * cdf_val.s2;
#if POINTS_EVAL >= 4
		final_val.s45 = w * cdf_val.s4;
		final_val.s67 = w * cdf_val.s6;
#endif
#endif
	}
	else if(is_integrand_pdf(stable->integrand))
	{
		final_val.s01 = w * pdf_val.s0;
#if POINTS_EVAL >= 2
		final_val.s23 = w * pdf_val.s2;
#if POINTS_EVAL >= 4
		final_val.s45 = w * pdf_val.s4;
		final_val.s67 = w * pdf_val.s6;
#endif
#endif
	}


	return final_val;
}


short precalculate_values(cl_precision x, constant struct stable_info* stable, struct stable_precalc* precalc)
{
	cl_precision x_, xxi, pdf_factor, cdf_factor, cdf_addition;

    x_ = (x - stable->mu_0) / stable->sigma;
   	xxi = x_ - stable->xi;

	precalc->iend = M_PI_2;

	pdf_factor = stable->final_pdf_factor;
	cdf_factor = stable->final_cdf_factor;
	cdf_addition = stable->final_cdf_addition;

	if(is_integrand_neq1(stable->integrand))
	{
		if (xxi < 0)
	    {
	        xxi = -xxi;
	        precalc->theta0_ = - stable->theta0;
	        precalc->beta_ = - stable->beta;

	        if(is_integrand_cdf(stable->integrand))
	       	{
	       		cdf_factor *= -1;

	       		if(stable->alfa < 1) // C1 changes here because of the sign inversion in θ0, recalculate
	       			cdf_addition = 1 - 0.5 + precalc->theta0_ * M_1_PI;
	       		else
	       			cdf_addition = 1 - stable->c1;
	       	}
	    }
	    else
		{
	    	precalc->theta0_ = stable->theta0;
	    	precalc->beta_ = stable->beta;
		}

	    if (xxi <= stable->xxi_th)
	    {
	       	precalc->pdf_precalc = stable->xi_coef * cos(stable->theta0) / stable->sigma;
	       	precalc->cdf_precalc = 0.5 - stable->theta0 * M_1_PI;

	        return SET_TO_RESULT_AND_RETURN;
	    }

		precalc->ibegin = - precalc->theta0_;

		precalc->xxipow = stable->alfainvalfa1 * log(fabs(xxi));

		if(is_integrand_pdf(stable->integrand))
			pdf_factor /= xxi;
	}
	else if(is_integrand_eq1(stable->integrand))
	{
		precalc->xxipow = (-M_PI * x_ * stable->c2_part);
		precalc->ibegin = - M_PI_2;
	}

	if (fabs(precalc->theta0_ + M_PI_2) < 2 * stable->THETA_TH)
	{
		precalc->pdf_precalc = 0;
		precalc->cdf_precalc = 0;
	    return SET_TO_RESULT_AND_RETURN;
	}

	precalc->xxi = xxi;

	if(is_integrand_pcdf(stable->integrand))
	{
		precalc->final_factor = (cl_precision2)(pdf_factor, cdf_factor);
		precalc->final_addition = (cl_precision2)(0, cdf_addition);
	}
	else if(is_integrand_pdf(stable->integrand))
	{
		precalc->final_factor = vec2(pdf_factor);
		precalc->final_addition = vec2(0);
	}
	else if(is_integrand_cdf(stable->integrand))
	{
		precalc->final_factor = vec2(cdf_factor);
		precalc->final_addition = vec2(cdf_addition);
	}

	return CONTINUE_CALC;
}

short scan_for_contributing_intervals(
#ifdef INTEL
	local cl_vec** sums
#else
	local cl_vec sums[MAX_WORKGROUPS][KRONROD_EVAL_POINTS]
#endif
	, local int* min_contributing, local int* max_contributing)
{
	size_t subinterval_index = get_local_id(1);
	size_t gk_point = get_local_id(0);

	if(gk_point == 0)
	{
		if(any(sums[subinterval_index][0].s01 >= SUBINT_CONTRIB_TH))
		{
			atomic_max(max_contributing, subinterval_index);
			atomic_min(min_contributing, subinterval_index);
		}

#if POINTS_EVAL >= 2
		if(any(sums[subinterval_index][0].s23 >= SUBINT_CONTRIB_TH))
		{
			atomic_max(max_contributing, subinterval_index + MAX_WORKGROUPS);
			atomic_min(min_contributing, subinterval_index + MAX_WORKGROUPS);
		}

#if POINTS_EVAL >= 4
		if(any(sums[subinterval_index][0].s45 >= SUBINT_CONTRIB_TH))
		{
			atomic_max(max_contributing, subinterval_index + MAX_WORKGROUPS * 2);
			atomic_min(min_contributing, subinterval_index + MAX_WORKGROUPS * 2);
		}

		if(any(sums[subinterval_index][0].s67 >= SUBINT_CONTRIB_TH))
		{
			atomic_max(max_contributing, subinterval_index + MAX_WORKGROUPS * 3);
			atomic_min(min_contributing, subinterval_index + MAX_WORKGROUPS * 3);
		}
#endif
#endif
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	int num_contributing = *max_contributing - *min_contributing + 1;

	return num_contributing > 0 && num_contributing < MIN_CONTRIBUTING_SUBINTS;
}

void calculate_integration_remainder(
#ifdef INTEL
	local cl_vec** sums
#else
	local cl_vec sums[MAX_WORKGROUPS][KRONROD_EVAL_POINTS]
#endif
	, struct stable_precalc* precalc, int min_contributing, int max_contributing, cl_precision2* previous_integration_remainder)
{
	size_t subinterval_index = get_local_id(1);
	size_t gk_point = get_local_id(0);
	size_t j;
	cl_precision2 current_integration_remainder = vec2(0);

	if(gk_point == 0 && subinterval_index == 0)
	{
		#pragma unroll
		for(j = 0; j < MAX_WORKGROUPS; j++)
		{
			if(j < min_contributing || j > max_contributing)
				current_integration_remainder += sums[j][0].s01;
#if POINTS_EVAL >= 2
			if(j + MAX_WORKGROUPS < min_contributing || j + MAX_WORKGROUPS > max_contributing)
				current_integration_remainder += sums[j][0].s23;
#if POINTS_EVAL >= 4
			if(j + 2 * MAX_WORKGROUPS < min_contributing || j + 2 * MAX_WORKGROUPS > max_contributing)
				current_integration_remainder += sums[j][0].s45;
			if(j + 3 * MAX_WORKGROUPS < min_contributing || j + 3 * MAX_WORKGROUPS > max_contributing)
				current_integration_remainder += sums[j][0].s67;
#endif
#endif
		}

		current_integration_remainder *= precalc->subint_length * precalc->final_factor;
		*previous_integration_remainder += current_integration_remainder;
	}
}


// If CDF or PDF: returns (kronrod, gauss).
// If PCDF: returns (PDF, CDF).
cl_precision2 stable_get_value(constant struct stable_info* stable, cl_precision x,
	#ifdef INTEL
	local cl_vec** sums
#else
	local cl_vec sums[MAX_WORKGROUPS][KRONROD_EVAL_POINTS]
#endif
	, local int* min_contributing, local int* max_contributing)
{
	size_t gk_point = get_local_id(0);
	size_t point_index = get_group_id(0);
	size_t subinterval_index = get_local_id(1);
	size_t points_count = get_num_groups(0);
	struct stable_precalc precalc;
	size_t offset;
	size_t j;
	short reevaluate = 0;
	cl_vec result = vec(0);
	size_t reevaluations = 0;
	cl_precision2 final;

	cl_precision2 previous_integration_remainder = vec2(0);

	*min_contributing = GK_SUBDIVISIONS;
	*max_contributing = 0;

	cl_precision pdf = 0;

   	if(precalculate_values(x, stable, &precalc) == SET_TO_RESULT_AND_RETURN)
	{
		// Return the precalculated values of the PDF and/or CDF. If we're on PCDF mode
		// return the PDF on the Gauss array and the CDF on the Kronrod array.
		final.y = is_integrand_pdf(stable->integrand) ? precalc.pdf_precalc : precalc.cdf_precalc;
		final.x = is_integrand_cdf(stable->integrand) ? precalc.cdf_precalc : precalc.pdf_precalc;

		return final;
	}

    do
    {
	    precalc.subint_length = (precalc.iend - precalc.ibegin) / GK_SUBDIVISIONS;

		offset = KRONROD_EVAL_POINTS / 2;

		if(gk_point < KRONROD_EVAL_POINTS)
		{
			result = eval_gk_pair(stable, &precalc);

			if(gk_point >= offset)
				sums[subinterval_index][gk_point] = result;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if(gk_point < offset)
			sums[subinterval_index][gk_point] = result + sums[subinterval_index][gk_point + offset];

		for(offset >>= 1; offset > 0; offset >>= 1)
		{
		    barrier(CLK_LOCAL_MEM_FENCE);

		    if (gk_point < offset)
		    	sums[subinterval_index][gk_point] += sums[subinterval_index][gk_point + offset];
		}

		reevaluations++;

		if(reevaluations > stable->max_reevaluations)
			break;

		if(stable->alfa <= 0.3)
		{
			// When alpha < 0.3, there's a big slope at the beginning of the subinterval
			// Reevaluate there to achieve more precision.
			*min_contributing = 0;
			*max_contributing = 0;

			reevaluate = 1;
		}
		else
		{
			reevaluate = scan_for_contributing_intervals(sums, min_contributing, max_contributing);
		}

		if(reevaluate)
		{
			int num_contributing = *max_contributing - *min_contributing + 1;

			calculate_integration_remainder(sums, &precalc, *min_contributing, *max_contributing, &previous_integration_remainder);

			precalc.ibegin = precalc.ibegin + *min_contributing * precalc.subint_length;
			precalc.iend = precalc.ibegin + num_contributing * precalc.subint_length;
		}
	} while(reevaluate);

	if(gk_point == 0 && subinterval_index == 0)
    {
    	// This is not a mistake: I measured it, this is faster than the other reduction.
    	// Probably due to the fact that for few workgroups the barriers and increased
    	// thread usage offset the advantage of the coalesced memory accesses.
    	cl_vec total = sums[0][0];

    	for(offset = 1; offset < MAX_WORKGROUPS; offset++)
    		total += sums[offset][0];

  		final = total.s01;
#if POINTS_EVAL >= 2
  		final += total.s23;
#if POINTS_EVAL >= 4
  		final += total.s45 + total.s67;
#endif
#endif
    	final *= precalc.subint_length * precalc.final_factor;

    	final += previous_integration_remainder + precalc.final_addition;
	}

	return final;
}

kernel void stable_points(constant struct stable_info* stable, constant cl_precision* x, global cl_precision* gauss, global cl_precision* kronrod)
{
	cl_precision2 val;
	size_t gk_point = get_local_id(0);
	size_t point_index = get_group_id(0);
	size_t subinterval_index = get_local_id(1);
	local cl_vec sums[MAX_WORKGROUPS][KRONROD_EVAL_POINTS];
	local int max_contributing, min_contributing;

	val = stable_get_value(stable, x[point_index], sums, &min_contributing, &max_contributing);

	if(gk_point == 0 && subinterval_index == 0)
	{
		gauss[point_index] = val.y;
		kronrod[point_index] = val.x;
	}
}

cl_precision stable_quick_inv_point(constant struct stable_info *stable, const cl_precision q, cl_precision *err)
{
	cl_precision x0 = 0;
	cl_precision C = 0;
	cl_precision alfa = stable->alfa;
	cl_precision beta = stable->beta;
	cl_precision q_ = q;
	cl_precision signBeta = 1;

	if (alfa < 0.1)   alfa = 0.1;

	if (beta < 0) {
		signBeta = -1;
		q_ = 1.0 - q_;
		beta = -beta;
	}
	if (beta == 1) {
		if (q_ < 0.1) {
			q_ = 0.1;
		}
	}

	/* Asympthotic expansion near the limits of the domain */
	if (q_ > 0.9 || q_ < 0.1) {
		if (alfa != 1.0)
			C = (1 - alfa) / (exp(lgamma(2 - alfa)) * cos(M_PI * alfa / 2.0));
		else
			C = 2 / M_PI;

		if (q_ > 0.9)
			x0 = pow((1 - q_) / (C * 0.5 * (1.0 + beta)), -1.0 / alfa);
		else
			x0 = -pow(q_ / (C * 0.5 * (1.0 - beta)), -1.0 / alfa);

		*err = 0.1;
	}

	else {
		/* Linear interpolation on precalculated values */
		int ia, ib, iq;
		cl_precision aux = 0;
		cl_precision xa = modf(alfa / 0.1, &aux); ia = (int)aux - 1;
		cl_precision xb = modf(beta / 0.2, &aux); ib = (int)aux;
		cl_precision xq = modf(  q_ / 0.1, &aux); iq = (int)aux - 1;

		if (alfa == 2) {ia = 18; xa = 1.0;}
		if (beta == 1) {ib = 4;  xb = 1.0;}
		if (q_ == 0.9) {iq = 7;  xq = 1.0;}

		cl_precision p[8] = {precalc[iq][ib][ia],   precalc[iq][ib][ia + 1],   precalc[iq][ib + 1][ia],   precalc[iq][ib + 1][ia + 1],
		               precalc[iq + 1][ib][ia], precalc[iq + 1][ib][ia + 1], precalc[iq + 1][ib + 1][ia], precalc[iq + 1][ib + 1][ia + 1]
		              };

		//Trilinear interpolation
		x0 = ((p[0] * (1.0 - xa) + p[1] * xa) * (1 - xb) + (p[2] * (1 - xa) + p[3] * xa) * xb) * (1 - xq) + ((p[4] * (1.0 - xa) + p[5] * xa) * (1 - xb) + (p[6] * (1 - xa) + p[7] * xa) * xb) * xq;

		if (err) {
			*err = fabs(0.5 * (p[0] - p[1]));
		}
	}

	x0 = x0 * signBeta * stable->sigma + stable->mu_0;

	return x0;
}

kernel void stable_quantile(constant struct stable_info* stable, constant cl_precision* q_vals, global cl_precision* err, global cl_precision* results)
{
	size_t gk_point = get_local_id(0);
	size_t point_index = get_group_id(0);
	size_t subinterval_index = get_local_id(1);
	cl_precision2 pcdf;
	cl_precision quantile = q_vals[point_index];
	cl_precision next_guess, error_priv;
	local cl_precision guess, error;
	size_t iterations = 0, max_iterations = 50;
	local cl_vec sums[MAX_WORKGROUPS][KRONROD_EVAL_POINTS];
	local int max_contributing, min_contributing;

	guess = stable_quick_inv_point(stable, quantile, &error_priv);
	error = error_priv;
	barrier(CLK_LOCAL_MEM_FENCE);

	// Newton method.
	while(error > stable->quantile_tolerance && iterations < max_iterations)
	{
		pcdf = stable_get_value(stable, guess, sums, &min_contributing, &max_contributing);

		if(gk_point == 0 && subinterval_index == 0)
		{
			cl_precision value = pcdf.y - quantile;
			cl_precision derivative = pcdf.x;

			if(fabs(value) < 1e-10)
			{
				error = 0;
			}
			else
			{
				next_guess = guess - value / derivative;
				error = fabs(next_guess - guess);
				guess = next_guess;
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
		iterations++;
	}

	if(gk_point == 0 && subinterval_index == 0)
	{
		results[point_index] = guess;
		err[point_index] = error;
	}
}

kernel void stable_rng(constant struct stable_info* stable, global cl_precision* results)
{
	size_t idx = get_local_id(0);

	results[idx] = idx;
}
