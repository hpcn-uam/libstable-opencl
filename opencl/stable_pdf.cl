#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#warning "Double precision floating point not supported by OpenCL implementation."
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

cl_vec eval_gk_pair(constant struct stable_info* stable, struct stable_precalc* precalc)
{
	size_t gk_point = get_local_id(0);
	size_t subinterval_index = get_local_id(1);

	cl_halfvec centers = vech(precalc->ibegin + precalc->subint_length / 2);
	cl_halfvec subd_offsets;
	cl_precision abscissa = precalc->subint_length * gk_absc[gk_point] / 2; // Translated integrand evaluation
	cl_precision2 abscissa_vec = (cl_precision2)(- abscissa, abscissa);
	cl_precision2 w = gk_weights[gk_point];

	subd_offsets = vech(subinterval_index);

#if POINTS_EVAL >= 2
	subd_offsets.s1 += 1 * GK_SUBDIVISIONS / POINTS_EVAL;
#if POINTS_EVAL >= 4
	subd_offsets.s2 += 2 * GK_SUBDIVISIONS / POINTS_EVAL;
	subd_offsets.s3 += 3 * GK_SUBDIVISIONS / POINTS_EVAL;
#endif
#endif

	subd_offsets *= precalc->subint_length;
	centers += subd_offsets;

	cl_vec val;

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

	cl_vec aux, aux2;
	cl_vec cosval = cos(val);
	cl_vec sinval = sin(val);

	if(stable->integrand == PDF_ALPHA_EQ1)
	{
		aux = (precalc->beta_ * val + vec(M_PI_2)) / cos(val);
		val = sin(val) * aux / precalc->beta_ + log(aux) + stable->k1;

		val = exp(val + precalc->xxipow);
		val = exp(-val) * val;
	}
	else if(stable->integrand == PDF_ALPHA_NEQ1)
	{
		aux = (precalc->theta0_ + val) * stable->alfa;
		val = log(cos(val) / sin(aux)) * stable->alfainvalfa1 +
			+ log(cos(aux - val) / cos(val)) + stable->k1;

		val = exp(val + precalc->xxipow);
		val = exp(-val) * val;
	}

	if(!isnormal(val.s0)) val.s0 = 0;
	if(!isnormal(val.s1)) val.s1 = 0;
#if POINTS_EVAL >= 2
	if(!isnormal(val.s2)) val.s2 = 0;
	if(!isnormal(val.s3)) val.s3 = 0;
#if POINTS_EVAL >= 4
	if(!isnormal(val.s4)) val.s4 = 0;
	if(!isnormal(val.s5)) val.s5 = 0;
	if(!isnormal(val.s6)) val.s6 = 0;
	if(!isnormal(val.s7)) val.s7 = 0;
#endif
#endif

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

	if(stable->integrand == PDF_ALPHA_NEQ1)
	{
		if (fabs(xxi) <= stable->xxi_th)
	    {
	        precalc->pdf_precalc = stable->xi_coef * cos(stable->theta0) / stable->sigma;
	        return SET_TO_RESULT_AND_RETURN;
	    }

	    precalc->theta0_ = stable->theta0;
	    precalc->beta_ = stable->beta;

	   	if (xxi < 0)
	    {
	        xxi = -xxi;
	        precalc->theta0_ = - precalc->theta0_;
	        precalc->beta_ = - precalc->beta_;
	    }

		precalc->ibegin = -precalc->theta0_;

		precalc->xxipow = stable->alfainvalfa1 * log(fabs(xxi));
	}
	else
	{
		precalc->ibegin = - M_PI_2;

		precalc->beta_ = fabs(stable->beta);
		precalc->xxipow = (-M_PI * x_ * stable->c2_part);
	}

	if (fabs(precalc->theta0_ + M_PI_2) < 2 * stable->THETA_TH)
	{
		precalc->pdf_precalc = 0;
	    return SET_TO_RESULT_AND_RETURN;
	}

	precalc->xxi = xxi;

	return CONTINUE_CALC;
}

kernel void stable_pdf_points(constant struct stable_info* stable, constant cl_precision* x, global cl_precision* gauss, global cl_precision* kronrod)
{
	size_t gk_point = get_local_id(0);
	size_t point_index = get_group_id(0);
	size_t subinterval_index = get_local_id(1);
	size_t points_count = get_num_groups(0);
	size_t offset_subinterval_index = subinterval_index + GK_SUBDIVISIONS / 2;
	struct stable_precalc precalc;
	size_t offset;
	size_t j;
	local cl_precision2 sums[GK_SUBDIVISIONS][KRONROD_EVAL_POINTS];
	local int min_contributing, max_contributing;
	short reevaluate = 0;

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

		if(gk_point < KRONROD_EVAL_POINTS)
		{
			cl_vec result = eval_gk_pair(stable, &precalc);

			sums[subinterval_index][gk_point] = result.s01;
#if POINTS_EVAL >= 2
			sums[subinterval_index + MAX_WORKGROUPS][gk_point] = result.s23;
#if POINTS_EVAL >= 4
			sums[subinterval_index + 2 * MAX_WORKGROUPS][gk_point] = result.s45;
			sums[subinterval_index + 3 * MAX_WORKGROUPS][gk_point] = result.s67;
#endif
#endif
		}

		for(offset = KRONROD_EVAL_POINTS / 2; offset > 0; offset >>= 1)
		{
		    barrier(CLK_LOCAL_MEM_FENCE);

		    if (gk_point < offset)
		    {
		  		for(j = 0; j < POINTS_EVAL; j++)
				{
					size_t subint_index = subinterval_index + MAX_WORKGROUPS * j;
					sums[subint_index][gk_point] += sums[subint_index][gk_point + offset];

					// Only check in the last iteration of the reduction loop.
					if(offset == 1 && gk_point == 0 && any(sums[subint_index][0] >= SUBINT_CONTRIB_TH))
					{
						atomic_max(&max_contributing, subint_index);
						atomic_min(&min_contributing, subint_index);
					}
				}
		  	}
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		int num_contributing = max_contributing - min_contributing + 1;

		if(!reevaluate && num_contributing > 0 && num_contributing < MIN_CONTRIBUTING_SUBINTS)
		{
			if(gk_point == 0 && subinterval_index == 0)
			{
				for(j = 0; j < GK_SUBDIVISIONS; j++)
				{
					if(j < min_contributing || j > max_contributing)
						previous_integration_remainder += sums[j][0];
				}

				previous_integration_remainder *= precalc.subint_length * stable->c2_part / stable->sigma;

				if(stable->integrand == PDF_ALPHA_NEQ1)
			    	previous_integration_remainder /= precalc.xxi;
			}

			precalc.ibegin = precalc.ibegin + min_contributing * precalc.subint_length;
			precalc.iend = precalc.ibegin + num_contributing * precalc.subint_length;

			reevaluate = 1;
		}
		else
		{
			reevaluate = 0;
		}

	} while(reevaluate);

#if POINTS_EVAL >= 4
	// We don't have enough threads to do the first iteration of the reduction as usual, so
	// we do it manually.
	sums[subinterval_index + MAX_WORKGROUPS][gk_point] +=
		sums[subinterval_index + MAX_WORKGROUPS * 2][gk_point] + sums[subinterval_index + MAX_WORKGROUPS * 3][gk_point];

	offset = GK_SUBDIVISIONS / 4;
#else
	// In this case, iterate as usual.
	offset = GK_SUBDIVISIONS / 2;
#endif

	for(; offset > 0; offset >>= 1)
	{
		if(subinterval_index < offset)
			sums[subinterval_index][gk_point] += sums[subinterval_index + offset][gk_point];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

    if(gk_point == 0 && subinterval_index == 0)
    {
    	sums[subinterval_index][gk_point] *= precalc.subint_length * stable->c2_part / stable->sigma;

    	if(stable->integrand == PDF_ALPHA_NEQ1)
	    	sums[subinterval_index][gk_point] /= precalc.xxi;

    	sums[subinterval_index][gk_point] += previous_integration_remainder;

		gauss[point_index] = sums[0][0].x;
		kronrod[point_index] = sums[0][0].y;
	}
}
