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
#define vec(b) (cl_precision2)((b), (b))
#define vec4(b) (cl_precision4)((b), (b), (b), (b))

#define SUBINT_CONTRIB_TH 0.00001
#define MIN_CONTRIBUTING_SUBINTS GK_SUBDIVISIONS / 4

cl_precision4 eval_gk_pair(constant struct stable_info* stable, struct stable_precalc* precalc, size_t subinterval_index, size_t gk_point)
{
	const cl_precision2 centers = vec(precalc->ibegin + precalc->half_subint_length) + precalc->subinterval_length * ((cl_precision2)(subinterval_index, subinterval_index + GK_SUBDIVISIONS / 2));
	const cl_precision abscissa = precalc->half_subint_length * gk_absc[gk_point]; // Translated integrand evaluation

	cl_precision4 val, res;
	cl_precision4 final_gk;
	cl_precision2 w = gk_weights[gk_point];
	val = (cl_precision4)(centers.x - abscissa, centers.x + abscissa, centers.y - abscissa, centers.y + abscissa);

	if(stable->integrand == PDF_ALPHA_EQ1)
	{
		cl_precision4 V, aux;

		aux = (precalc->beta_ * val + vec4(M_PI_2)) / cos(val);
		V = sin(val) * aux / precalc->beta_ + log(aux) + stable->k1;

		res = exp(V + precalc->xxipow);
		res = exp(-res) * res;
	}
	else if(stable->integrand == PDF_ALPHA_NEQ1)
	{
		cl_precision4 cos_theta, aux, V;

		cos_theta = cos(val);

		aux = (precalc->theta0_ + val) * stable->alfa;
		V = log(cos_theta / sin(aux)) * stable->alfainvalfa1 +
			+ log(cos(aux - val) / cos_theta) + stable->k1;

		res = exp(V + precalc->xxipow);
		res = exp(-res) * res;
	}
	else if(stable->integrand == GPU_TEST_INTEGRAND)
	{
		res = (val + 4) * (val - 3) * (val + 0) * (val + 1) * (val + 4) * (val - 3) * (val + 0) * (val + 1);
	}
	else if(stable->integrand == GPU_TEST_INTEGRAND_SIMPLE)
	{
		res = val;
	}

	if(!isnormal(res.x))
		res.x = 0;

	if(!isnormal(res.y))
		res.y = 0;

	if(!isnormal(res.z))
		res.z = 0;

	if(!isnormal(res.w))
		res.w = 0;

	if(gk_point < KRONROD_EVAL_POINTS - 1)
	{
		res.x += res.y;
		res.z += res.w;
	}

	final_gk.xy = w * res.x;
	final_gk.zw = w * res.z;

	return final_gk;
}

kernel void stable_pdf_points(constant struct stable_info* stable, constant cl_precision* x, global cl_precision* gauss, global cl_precision* kronrod)
{
	size_t gk_point = get_local_id(0);
	size_t point_index = get_group_id(0);
	size_t subinterval_index = get_local_id(1);
	size_t points_count = get_num_groups(0);
	size_t half_subinterval_count = GK_SUBDIVISIONS / 2;
	size_t offset_subinterval_index = subinterval_index + GK_SUBDIVISIONS / 2;
	struct stable_precalc precalc;
	size_t offset;
	size_t j;
	local cl_precision2 sums[GK_SUBDIVISIONS][KRONROD_EVAL_POINTS];
	local int min_contributing, max_contributing;
	short reevaluate = 0;
	cl_precision2 previous_integration_remainder = vec(0);

	min_contributing = GK_SUBDIVISIONS;
	max_contributing = 0;

	cl_precision pdf = 0;
    cl_precision x_, xxi;


    x_ = (x[point_index] - stable->mu_0) / stable->sigma;
   	xxi = x_ - stable->xi;

    if(stable->integrand == PDF_ALPHA_NEQ1)
    {
    	if (fabs(xxi) <= stable->xxi_th)
	    {
	        pdf = stable->xi_coef * cos(stable->theta0);

	        gauss[point_index] = pdf / stable->sigma;
	        kronrod[point_index] = pdf / stable->sigma;
	        return;
	    }

	   	if (xxi < 0)
	    {
	        xxi = -xxi;
	        precalc.theta0_ = -stable->theta0;
	        precalc.beta_ = -stable->beta;
	    }
	    else
	    {
	        precalc.theta0_ = stable->theta0;
	        precalc.beta_ = stable->beta;
	    }

    	precalc.ibegin = -precalc.theta0_;
    	precalc.iend = M_PI_2;
	}
	else
	{
    	precalc.ibegin = - M_PI_2;
    	precalc.iend = M_PI_2;

    	precalc.beta_ = fabs(stable->beta);
	}

	if(stable->integrand == PDF_ALPHA_NEQ1)
	    precalc.xxipow = stable->alfainvalfa1 * log(fabs(xxi));
    else
		precalc.xxipow = (-M_PI * x_ * stable->c2_part);

    if (fabs(precalc.theta0_ + M_PI_2) < 2 * stable->THETA_TH)
    {
    	gauss[point_index] = 0;
        kronrod[point_index] = 0;
        return;
    }

    do
    {
	    precalc.subinterval_length = (precalc.iend - precalc.ibegin) / (half_subinterval_count * 2);
	    precalc.half_subint_length = precalc.subinterval_length / 2;

		if(gk_point < KRONROD_EVAL_POINTS)
		{
			cl_precision4 result = eval_gk_pair(stable, &precalc, subinterval_index, gk_point);
			sums[subinterval_index][gk_point] = result.xy;
			sums[offset_subinterval_index][gk_point] = result.zw;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		for(offset = KRONROD_EVAL_POINTS / 2; offset > 0; offset >>= 1)
		{
		    if (gk_point < offset)
		    {
		  		sums[subinterval_index][gk_point] += sums[subinterval_index][gk_point + offset];
		  		sums[offset_subinterval_index][gk_point] += sums[offset_subinterval_index][gk_point + offset];
		  	}

		    barrier(CLK_LOCAL_MEM_FENCE);
		}

		if(gk_point == 0)
		{
			if(any(sums[subinterval_index][gk_point] >= SUBINT_CONTRIB_TH))
			{
				atomic_max(&max_contributing, subinterval_index);
				atomic_min(&min_contributing, subinterval_index);
			}

			if(any(sums[offset_subinterval_index][gk_point] >= SUBINT_CONTRIB_TH))
			{
				atomic_max(&max_contributing, offset_subinterval_index);
				atomic_min(&min_contributing, offset_subinterval_index);
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

				if(stable->integrand == PDF_ALPHA_NEQ1)
			    	previous_integration_remainder *= precalc.subinterval_length * stable->c2_part / (xxi * stable->sigma);
				else
					previous_integration_remainder *= precalc.subinterval_length * stable->c2_part / stable->sigma;
			}

			precalc.ibegin = precalc.ibegin + min_contributing * precalc.subinterval_length;
			precalc.iend = precalc.ibegin + num_contributing * precalc.subinterval_length;

			reevaluate = 1;
		}
		else
		{
			reevaluate = 0;
		}

	} while(reevaluate);

	for(offset = half_subinterval_count; offset > 0; offset >>= 1)
	{
		if(subinterval_index < offset)
			sums[subinterval_index][gk_point] += sums[subinterval_index + offset][gk_point];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

    if(gk_point == 0 && subinterval_index == 0)
    {
    	if(stable->integrand == PDF_ALPHA_NEQ1)
	    	sums[0][0] *= precalc.subinterval_length * stable->c2_part / (xxi * stable->sigma);
    	else
    		sums[0][0] *= precalc.subinterval_length * stable->c2_part / stable->sigma;

    	sums[0][0] += previous_integration_remainder;

		gauss[point_index] = sums[0][0].x;
		kronrod[point_index] = sums[0][0].y;
	}
}
