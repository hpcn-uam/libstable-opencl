#ifndef OPENCL_COMMON_H
#define OPENCL_COMMON_H

#define GK_USE_127_POINTS

#ifdef GK_USE_127_POINTS
#define GK_POINTS 127
#endif

#ifdef GK_USE_61_POINTS
#define GK_POINTS 61
#endif

#define GK_SUBDIVISIONS 8
#define KRONROD_EVAL_POINTS (GK_POINTS / 2 + 1)

#define PDF_ALPHA_EQ1 1
#define PDF_ALPHA_NEQ1 2
#define GPU_TEST_INTEGRAND 100
#define GPU_TEST_INTEGRAND_SIMPLE 101

#if defined(FLOAT_GPU_UNIT) || (defined(__OPENCL_VERSION__) && !defined(cl_khr_fp64) && !defined(cl_amd_fp64))
#define cl_precision float
#define cl_precision2 float2
#define cl_precision_type "float"
#define CL_PRECISION_IS_FLOAT
#else
#define cl_precision double
#define cl_precision2 double2
#define cl_precision_type "double"
#endif


struct stable_info {
    cl_precision theta0_;
    cl_precision beta_;
    cl_precision xxipow;
    cl_precision ibegin;
    cl_precision iend;
    cl_precision theta;
    cl_precision k1;
    cl_precision theta0;
    cl_precision alfa;
    cl_precision alfainvalfa1;
    cl_precision mu_0;
    cl_precision sigma;
    cl_precision xi;
    cl_precision xxi_th;
    cl_precision S;
    cl_precision c2_part;
    cl_precision THETA_TH;
    cl_precision beta;
    unsigned int threads_per_interval;
    unsigned int gauss_points;
    unsigned int kronrod_points;
    unsigned int integrand;
};

struct stable_precalc {
    cl_precision theta0_;
    cl_precision beta_;
    cl_precision xxipow;
    cl_precision ibegin;
    cl_precision iend;
    cl_precision subinterval_length;
    cl_precision half_subint_length;
};

#endif
