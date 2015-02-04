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

cl_precision stable_pdf_g1(cl_precision theta, constant struct stable_info* stable);

#define anyf(a) any((int2) a)
#define vec(b) (cl_precision2)((b), (b))

cl_precision gammaln(cl_precision xx)
{
	int j;
	cl_precision x, y, tmp, ser;
	const cl_precision cof[6]= {76.18009172947146, -86.50532032941677,
		24.01409824083091, -1.231739572450155, 0.1208650973866179e-2,
		-0.5395239384953e-5};

	y=x=xx;
	tmp=x+5.5;
	tmp-=(x+0.5)*log(tmp);
	ser=1.000000000190015;

	for (j=0;j<6;j++)
	{
		y += 1;
		ser += cof[j]/ y;
	}

	return -tmp+log(2.5066282746310005*ser/x);

}


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

cl_precision2 eval_gk_pair(constant struct stable_info* stable, struct stable_precalc* precalc, size_t subinterval_index, size_t gk_point)
{
	const cl_precision center = precalc->ibegin + stable->subinterval_length * subinterval_index + stable->half_subint_length;
	const cl_precision abscissa = stable->half_subint_length * gk_absc[gk_point]; // Translated integrand evaluation

	cl_precision2 val, res;
	cl_precision2 w = gk_weights[gk_point];
	val = (cl_precision2)(center - abscissa, center + abscissa);

	if(stable->integrand == PDF_ALPHA_EQ1)
	{
		cl_precision2 V, aux;

		aux = (precalc->beta_ * val + (cl_precision2)(M_PI_2, M_PI_2)) / cos(val);
		V = sin(val) * aux / precalc->beta_ + log(aux) + stable->k1;

		res = exp(V + precalc->xxipow);
		res = exp(-res) * res;
	}
	else if(stable->integrand == PDF_ALPHA_NEQ1)
	{
		cl_precision2 cos_theta, aux, V;

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

	if(gk_point < KRONROD_EVAL_POINTS - 1)
		res.x += res.y;

	return w * res.x;
}

void _stable_pdf_integ(global cl_precision* gauss, global cl_precision* kronrod, constant struct stable_info* stable, struct stable_precalc* precalc, local cl_precision2* sums)
{
	size_t gk_point = get_local_id(0);
	size_t subinterval_index = get_group_id(0);
	size_t subinterval_count = get_local_size(0);
	size_t interval_count = get_local_size(1);
	size_t i;
	cl_precision gauss_sum = 0, kronrod_sum = 0;

	if(gk_point < KRONROD_EVAL_POINTS)
		sums[gk_point] = eval_gk_pair(stable, precalc, subinterval_index, gk_point);

	barrier(CLK_LOCAL_MEM_FENCE);

	for(size_t offset = KRONROD_EVAL_POINTS / 2; offset > 0; offset >>= 1)
	{
	    if (gk_point < offset)
	  		sums[gk_point] += sums[gk_point + offset];

	    barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(gk_point == 0)
	{
		gauss[subinterval_index] = sums[0].y * stable->subinterval_length;
		kronrod[subinterval_index] = sums[0].x * stable->subinterval_length;
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if(subinterval_index == 0)
	{
		size_t offset = subinterval_count / 2;

		if(gk_point < offset)
		{
			sums[gk_point].x = gauss[gk_point + offset] + gauss[gk_point];
			sums[gk_point].y = kronrod[gk_point + offset] + kronrod[gk_point];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		for(offset = offset >> 1; offset > 0; offset >>= 1)
		{
			if(gk_point < offset)
				sums[gk_point] += sums[gk_point + offset];

			barrier(CLK_LOCAL_MEM_FENCE);
		}

		if(gk_point == 0)
		{
			gauss[0] = sums[0].x;
			kronrod[0] = sums[0].y;
		}
	}
}

kernel void stable_pdf_integ(global cl_precision* gauss, global cl_precision* kronrod, constant struct stable_info* stable)
{
	struct stable_precalc precalc;
	local cl_precision2 sums[GK_POINTS / 2 + 1];

	precalc.theta0_ = stable->theta0_;
   	precalc.beta_ = stable->beta_;
   	precalc.xxipow = stable->xxipow;
   	precalc.ibegin = stable->ibegin;
   	precalc.iend = stable->iend;

	_stable_pdf_integ(gauss, kronrod, stable, &precalc, sums);
}

kernel void stable_pdf_points(constant struct stable_info* stable, constant cl_precision* x, global cl_precision* gauss, global cl_precision* kronrod)
{
	size_t gk_point = get_local_id(0);
	size_t point_index = get_group_id(1);
	size_t subinterval_index = get_group_id(0);
	size_t interval_count = get_num_groups(1);
	size_t subinterval_count = get_num_groups(0);
	struct stable_precalc precalc;
	local cl_precision2 sums[GK_POINTS / 2 + 1];

	cl_precision pdf = 0;
    cl_precision x_, xxi;
 	cl_precision ibegin, iend;

    x_ = (x[point_index] - stable->mu_0) / stable->sigma;
   	xxi = x_ - stable->xi;

    if (fabs(xxi) <= stable->xxi_th)
    {
        pdf = exp(gammaln(1.0 + 1.0 / stable->alfa)) *
              cos(stable->theta0) / (M_PI * stable->S);

        gauss[point_index * interval_count] = pdf / stable->sigma;
        kronrod[point_index * interval_count] = pdf / stable->sigma;
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

    precalc.ibegin = -precalc.theta0_ + stable->THETA_TH;;
    precalc.iend = M_PI_2 - stable->THETA_TH;

    precalc.xxipow = stable->alfainvalfa1 * log(fabs(xxi));

    if (fabs(precalc.theta0_ + M_PI_2) < 2 * stable->THETA_TH)
    {
    	gauss[point_index * interval_count] = 0;
        kronrod[point_index * interval_count] = 0;
        return;
    }

    _stable_pdf_integ(gauss + point_index * interval_count, kronrod + point_index * interval_count, stable, &precalc, sums);

    pdf = stable->c2_part / xxi * pdf;
    pdf = pdf / stable->sigma;
}
