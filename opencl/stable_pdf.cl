#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#warning "Double precision floating point not supported by OpenCL implementation."
#endif

#ifndef M_PI_2
#define M_PI_2     1.57079632679489661923132169164      // Pi/2
#endif

#include "includes/opencl_common.h"
#include "includes/gk_points.h"

cl_precision stable_pdf_g1(cl_precision theta, constant struct stable_info* stable);

#define anyf(a) any((int2) a)
#define vec(b) (cl_precision2)((b), (b))

cl_precision stable_pdf_alpha_neq1(cl_precision theta, constant struct stable_info *args)
{
	cl_precision g, cos_theta, aux, V;

	cos_theta = cos(theta);
	aux = (args->theta0_ + theta) * args->alfa;
	V = log(cos_theta / sin(aux)) * args->alfainvalfa1 +
	+ log(cos(aux - theta) / cos_theta) + args->k1;

	g = V + args->xxipow;
	if (g > 6.55 || g < -700) return 0.0;
	else  g = exp(g);
	g = exp(-g) * g;
	if (isnan(g) || isinf(g) || g < 0)
	{
		return 0.0;
	}

	return g;
}

cl_precision stable_pdf_alpha_eq1(cl_precision theta, constant struct stable_info* stable)
{
	cl_precision g, V, aux;

	aux = (stable->beta_ * theta + M_PI_2) / cos(theta);
	V = sin(theta) * aux / stable->beta_ + log(aux) + stable->k1;

	g = V + stable->xxipow;
	if(isnan(g)) return 0.0;
	if ((g = exp(g)) < 1.522e-8 ) return (1.0 - g) * g;
	g = exp(-g) * g;
	if (isnan(g) || g < 0) return 0.0;

	return g;
}

kernel void stable_pdf(global cl_precision* gauss, global cl_precision* kronrod, constant struct stable_info* stable)
{
	size_t subinterval_index = get_local_id(0);
	size_t interval = get_group_id(0);

	const int kronrod_eval_points = GK_POINTS / 2 + 1;
	local cl_precision2 sums[GK_POINTS / 2 + 1];

	if(subinterval_index < kronrod_eval_points)
	{
		const cl_precision center = stable->ibegin + stable->subinterval_length * interval + stable->half_subint_length;
		const cl_precision abscissa = stable->half_subint_length * gk_absc[subinterval_index]; // Translated integrand evaluation

		cl_precision2 val, res;
		cl_precision2 w = gk_weights[subinterval_index];
		val = (cl_precision2)(center - abscissa, center + abscissa);

		if(stable->integrand == PDF_ALPHA_EQ1)
		{
			cl_precision2 V, aux;

			aux = (stable->beta_ * val + (cl_precision2)(M_PI_2, M_PI_2)) / cos(val);
			V = sin(val) * aux / stable->beta_ + log(aux) + stable->k1;

			res = exp(V + stable->xxipow);
			res = exp(-res) * res;
		}
		else if(stable->integrand == PDF_ALPHA_NEQ1)
		{
			cl_precision2 cos_theta, aux, V;

			cos_theta = cos(val);

			aux = (stable->theta0_ + val) * stable->alfa;
			V = log(cos_theta / sin(aux)) * stable->alfainvalfa1 +
				+ log(cos(aux - val) / cos_theta) + stable->k1;

			res = exp(V + stable->xxipow);
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

calcend: // Sorry.
		if(!isnormal(res.x))
			res.x = 0;

		if(!isnormal(res.y))
			res.y = 0;

		if(subinterval_index < kronrod_eval_points - 1)
			res.x += res.y;

		sums[subinterval_index] = w * res.x;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for(size_t offset = kronrod_eval_points / 2; offset > 0; offset >>= 1)
	{
	    if (subinterval_index < offset)
	  		sums[subinterval_index] += sums[subinterval_index + offset];

	    barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(subinterval_index == 0)
	{
		gauss[interval] = sums[0].y * stable->subinterval_length;
		kronrod[interval] = sums[0].x * stable->subinterval_length;
	}
}
