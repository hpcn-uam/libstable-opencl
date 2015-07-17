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
#define M_PI       3.14159265358979323846264338327950288       // Pi/2
#endif

#include "includes/opencl_common.h"
#include "includes/gk_points.h"

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

	cl_vec val;

	// Calculate the points of evaluation: add each center to our two abscissae.
#if POINTS_EVAL == 1
	val.s01 = vec2(centers) + abscissa_vec;
#else
	val.s01 = centers.s00 + abscissa_vec;
#if POINTS_EVAL >= 2
	val.s23 = centers.s11 + abscissa_vec;
#if POINTS_EVAL >= 4
	val.s45 = centers.s22 + abscissa_vec;
	val.s67 = centers.s33 + abscissa_vec;
#endif
#endif
#endif

	// Now evaluate the function.
	cl_vec aux;
	cl_vec cosval = cos(val);

	if(stable->integrand == PDF_ALPHA_EQ1)
	{
		aux = (precalc->beta_ * val + vec(M_PI_2)) / cosval;
		val = sin(val) * aux / precalc->beta_ + log(aux) + stable->k1;

		val = exp(val + precalc->xxipow);
		val = exp(-val) * val;
	}
	else if(stable->integrand == PDF_ALPHA_NEQ1)
	{
		aux = (precalc->theta0_ + val) * stable->alfa;
		val = log(cosval / sin(aux)) * stable->alfainvalfa1 +
			+ log(cos(aux - val) / cosval) + stable->k1;

		val = exp(val + precalc->xxipow);
		val = exp(-val) * val;
	}
	else if(stable->integrand == CDF_ALPHA_NEQ1)
	{
		aux = (precalc->theta0_ + val) * stable->alfa;
		val = log(cosval / sin(aux)) * stable->alfainvalfa1 +
			+ log(cos(aux - val) / cosval) + stable->k1;

		val = exp(-exp(val + precalc->xxipow));
	}
	else if(stable->integrand == CDF_ALPHA_EQ1)
	{
		aux = (precalc->beta_ * val + vec(M_PI_2)) / cosval;
		val = sin(val) * aux / precalc->beta_ + log(aux) + stable->k1;

		val = exp(-exp(val + precalc->xxipow));
	}

	// GK quadrature is symmetric, so just add them in one quantity.
	// Just avoid the 0 (last evaluation point) because it's the only
	// point being evaluated once.
	if(gk_point < KRONROD_EVAL_POINTS - 1)
	{
		val.s0 += val.s1;
#if POINTS_EVAL >= 2
		val.s2 += val.s3;
#if POINTS_EVAL >= 4
		val.s4 += val.s5;
		val.s6 += val.s7;
#endif
#endif
	}

	// Now multiply the result by the corresponding weight and return.
	val.s01 = w * val.s0;
#if POINTS_EVAL >= 2
	val.s23 = w * val.s2;
#if POINTS_EVAL >= 4
	val.s45 = w * val.s4;
	val.s67 = w * val.s6;
#endif
#endif


	return val;
}


short precalculate_values(cl_precision x, constant struct stable_info* stable, struct stable_precalc* precalc)
{
	cl_precision x_, xxi;

    x_ = (x - stable->mu_0) / stable->sigma;
   	xxi = x_ - stable->xi;

	precalc->iend = M_PI_2;

	precalc->final_factor = stable->final_factor;
	precalc->final_addition = stable->final_addition;

	if(is_integrand_neq1(stable->integrand))
	{
		if (xxi < 0)
	    {
	        xxi = -xxi;
	        precalc->theta0_ = - stable->theta0;
	        precalc->beta_ = - stable->beta;

	        if(stable->integrand == CDF_ALPHA_NEQ1)
	       	{
	       		precalc->final_factor *= -1;
	       		precalc->final_addition = 1 - stable->c1;
	       	}
	    }
	    else
		{
	    	precalc->theta0_ = stable->theta0;
	    	precalc->beta_ = stable->beta;
		}

	    if (xxi <= stable->xxi_th)
	    {
	    	if(stable->integrand == PDF_ALPHA_NEQ1)
	        	precalc->pdf_precalc = stable->xi_coef * cos(stable->theta0) / stable->sigma;
	        else // CDF_ALPHA_NEQ1
	        	precalc->pdf_precalc = 0.5 - stable->theta0 * M_1_PI;

	        return SET_TO_RESULT_AND_RETURN;
	    }

		precalc->ibegin = - precalc->theta0_;

		precalc->xxipow = stable->alfainvalfa1 * log(fabs(xxi));

		if(stable->integrand == PDF_ALPHA_NEQ1)
			precalc->final_factor /= xxi;
	}
	else if(is_integrand_eq1(stable->integrand))
	{
		precalc->xxipow = (-M_PI * x_ * stable->c2_part);
		precalc->ibegin = - M_PI_2;
	}

	if (fabs(precalc->theta0_ + M_PI_2) < 2 * stable->THETA_TH)
	{
		precalc->pdf_precalc = 0;
	    return SET_TO_RESULT_AND_RETURN;
	}

	precalc->xxi = xxi;

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

kernel void stable_points(constant struct stable_info* stable, constant cl_precision* x, global cl_precision* gauss, global cl_precision* kronrod)
{
	size_t gk_point = get_local_id(0);
	size_t point_index = get_group_id(0);
	size_t subinterval_index = get_local_id(1);
	size_t points_count = get_num_groups(0);
	struct stable_precalc precalc;
	size_t offset;
	size_t j;
	local cl_vec sums[MAX_WORKGROUPS][KRONROD_EVAL_POINTS];
	local int min_contributing, max_contributing;
	short reevaluate = 0;
	cl_vec result = vec(0);
	size_t reevaluations = 0;

	cl_precision2 previous_integration_remainder = vec2(0);

	min_contributing = GK_SUBDIVISIONS;
	max_contributing = 0;

	cl_precision pdf = 0;

   	if(precalculate_values(x[point_index], stable, &precalc) == SET_TO_RESULT_AND_RETURN)
	{
		gauss[point_index] = precalc.pdf_precalc;
		kronrod[point_index] = precalc.pdf_precalc;
		return;
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
			min_contributing = 0;
			max_contributing = 0;

			reevaluate = 1;
		}
		else
		{
			reevaluate = scan_for_contributing_intervals(sums, &min_contributing, &max_contributing);
		}

		if(reevaluate)
		{
			int num_contributing = max_contributing - min_contributing + 1;

			calculate_integration_remainder(sums, &precalc, min_contributing, max_contributing, &previous_integration_remainder);

			precalc.ibegin = precalc.ibegin + min_contributing * precalc.subint_length;
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

  		cl_precision2 final = total.s01;
#if POINTS_EVAL >= 2
  		final += total.s23;
#if POINTS_EVAL >= 4
  		final += total.s45 + total.s67;
#endif
#endif

    	final *= precalc.subint_length * precalc.final_factor;

    	final += previous_integration_remainder + precalc.final_addition;

		gauss[point_index] = final.y;
		kronrod[point_index] = final.x;
	}
}
